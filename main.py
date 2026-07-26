import os
import pickle
from typing import Optional, List, Dict, Any, Tuple
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# =========================
# ENV
# =========================
load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

TMDB_BASE = "https://api.themoviedb.org/3"
TMDB_IMG_500 = "https://image.tmdb.org/t/p/w500"

if not TMDB_API_KEY:
    raise RuntimeError("TMDB_API_KEY missing. Put it in .env")


# =========================
# GLOBALS
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DF_PATH = os.path.join(BASE_DIR, "df.pkl")
INDICES_PATH = os.path.join(BASE_DIR, "indices.pkl")
TFIDF_MATRIX_PATH = os.path.join(BASE_DIR, "tfidf_matrix.pkl")
TFIDF_PATH = os.path.join(BASE_DIR, "tfidf.pkl")

df: Optional[pd.DataFrame] = None
indices_obj: Any = None
tfidf_matrix: Any = None
tfidf_obj: Any = None
TITLE_TO_IDX: Optional[Dict[str, int]] = None

client: Optional[httpx.AsyncClient] = None


# =========================
# LIFESPAN (FIXED)
# =========================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global df, indices_obj, tfidf_matrix, tfidf_obj, TITLE_TO_IDX, client

    # Load pickles
    with open(DF_PATH, "rb") as f:
        df = pickle.load(f)

    with open(INDICES_PATH, "rb") as f:
        indices_obj = pickle.load(f)

    with open(TFIDF_MATRIX_PATH, "rb") as f:
        tfidf_matrix = pickle.load(f)

    with open(TFIDF_PATH, "rb") as f:
        tfidf_obj = pickle.load(f)

    TITLE_TO_IDX = build_title_to_idx_map(indices_obj)

    if df is None or "title" not in df.columns:
        raise RuntimeError("df.pkl must contain 'title' column")

    # Create shared HTTP client
    client = httpx.AsyncClient(timeout=20)

    print("✅ App started")

    yield

    # Cleanup
    await client.aclose()
    print("🛑 App stopped")


# =========================
# FASTAPI APP
# =========================
app = FastAPI(
    title="Movie Recommender API",
    version="3.1",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# MODELS
# =========================
class TMDBMovieCard(BaseModel):
    tmdb_id: int
    title: str
    poster_url: Optional[str] = None
    release_date: Optional[str] = None
    vote_average: Optional[float] = None


class TMDBMovieDetails(BaseModel):
    tmdb_id: int
    title: str
    overview: Optional[str] = None
    release_date: Optional[str] = None
    poster_url: Optional[str] = None
    backdrop_url: Optional[str] = None
    genres: List[dict] = Field(default_factory=list)


class TFIDFRecItem(BaseModel):
    title: str
    score: float
    tmdb: Optional[TMDBMovieCard] = None


class SearchBundleResponse(BaseModel):
    query: str
    movie_details: TMDBMovieDetails
    tfidf_recommendations: List[TFIDFRecItem]
    genre_recommendations: List[TMDBMovieCard]


# =========================
# UTILS
# =========================
def _norm_title(t: str) -> str:
    return str(t).strip().lower()


def make_img_url(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    return f"{TMDB_IMG_500}{path}"


async def tmdb_get(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    global client

    q = dict(params)
    q["api_key"] = TMDB_API_KEY

    try:
        r = await client.get(f"{TMDB_BASE}{path}", params=q)
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=502,
            detail=f"TMDB request error: {repr(e)}",
        )

    if r.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"TMDB error {r.status_code}: {r.text}",
        )

    return r.json()


async def tmdb_cards_from_results(results: List[dict], limit: int = 20):
    return [
        TMDBMovieCard(
            tmdb_id=int(m["id"]),
            title=m.get("title") or "",
            poster_url=make_img_url(m.get("poster_path")),
            release_date=m.get("release_date"),
            vote_average=m.get("vote_average"),
        )
        for m in (results or [])[:limit]
    ]


async def tmdb_movie_details(movie_id: int) -> TMDBMovieDetails:
    data = await tmdb_get(f"/movie/{movie_id}", {"language": "en-US"})
    return TMDBMovieDetails(
        tmdb_id=int(data["id"]),
        title=data.get("title") or "",
        overview=data.get("overview"),
        release_date=data.get("release_date"),
        poster_url=make_img_url(data.get("poster_path")),
        backdrop_url=make_img_url(data.get("backdrop_path")),
        genres=data.get("genres", []),
    )


async def tmdb_search_movies(query: str, page: int = 1):
    return await tmdb_get(
        "/search/movie",
        {"query": query, "include_adult": "false", "language": "en-US", "page": page},
    )


async def tmdb_search_first(query: str):
    data = await tmdb_search_movies(query, 1)
    return data.get("results", [None])[0]


# =========================
# TF-IDF
# =========================
def build_title_to_idx_map(indices: Any) -> Dict[str, int]:
    title_to_idx = {}
    for k, v in indices.items():
        title_to_idx[_norm_title(k)] = int(v)
    return title_to_idx


def get_local_idx_by_title(title: str) -> int:
    key = _norm_title(title)
    if key in TITLE_TO_IDX:
        return TITLE_TO_IDX[key]
    raise HTTPException(404, f"Title not found: {title}")


def tfidf_recommend_titles(query_title: str, top_n: int = 10):
    idx = get_local_idx_by_title(query_title)
    qv = tfidf_matrix[idx]
    scores = (tfidf_matrix @ qv.T).toarray().ravel()
    order = np.argsort(-scores)

    out = []
    for i in order:
        if i == idx:
            continue
        out.append((df.iloc[i]["title"], float(scores[i])))
        if len(out) >= top_n:
            break
    return out


async def attach_tmdb_card_by_title(title: str):
    try:
        m = await tmdb_search_first(title)
        if not m:
            return None
        return TMDBMovieCard(
            tmdb_id=m["id"],
            title=m.get("title"),
            poster_url=make_img_url(m.get("poster_path")),
            release_date=m.get("release_date"),
            vote_average=m.get("vote_average"),
        )
    except:
        return None


# =========================
# ROUTES
# =========================
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/home", response_model=List[TMDBMovieCard])
async def home(
    category: str = Query("popular"),
    limit: int = Query(24, ge=1, le=50),
):

    try:

        # ✅ TRENDING SPECIAL CASE
        if category == "trending":

            data = await tmdb_get(
                "/trending/movie/day",
                {"language": "en-US"}
            )

            return await tmdb_cards_from_results(
                data.get("results", []),
                limit=limit
            )

        # ✅ NORMAL MOVIE CATEGORIES
        if category not in {
            "popular",
            "top_rated",
            "upcoming",
            "now_playing",
        }:
            raise HTTPException(
                status_code=400,
                detail="Invalid category"
            )

        data = await tmdb_get(
            f"/movie/{category}",
            {
                "language": "en-US",
                "page": 1
            }
        )

        return await tmdb_cards_from_results(
            data.get("results", []),
            limit=limit
        )

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Home route failed: {e}"
        )


@app.get("/tmdb/search")
async def tmdb_search(query: str, page: int = 1):
    return await tmdb_search_movies(query, page)


@app.get("/movie/id/{tmdb_id}")
async def movie_details_route(tmdb_id: int):
    return await tmdb_movie_details(tmdb_id)


@app.get("/recommend/tfidf")
async def recommend_tfidf(title: str, top_n: int = 10):
    recs = tfidf_recommend_titles(title, top_n)
    return [{"title": t, "score": s} for t, s in recs]


@app.get("/movie/search")
async def search_bundle(query: str):
    best = await tmdb_search_first(query)
    if not best:
        raise HTTPException(404, "Movie not found")

    details = await tmdb_movie_details(best["id"])
    recs = tfidf_recommend_titles(details.title, 10)

    tfidf_items = []
    for t, s in recs:
        card = await attach_tmdb_card_by_title(t)
        tfidf_items.append({"title": t, "score": s, "tmdb": card})

    return {
        "query": query,
        "movie_details": details,
        "tfidf_recommendations": tfidf_items,
    }

@app.get("/recommend/similar", response_model=List[TMDBMovieCard])
async def recommend_similar(
    tmdb_id: int = Query(...),
    limit: int = Query(18, ge=1, le=50),
):

    data = await tmdb_get(
        f"/movie/{tmdb_id}/similar",
        {
            "language": "en-US",
            "page": 1,
        }
    )

    return await tmdb_cards_from_results(
        data.get("results", []),
        limit=limit,
    )
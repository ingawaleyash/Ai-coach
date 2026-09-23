import os
import logging
import uuid

from django.conf import settings
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages

from .models import UploadedVideo, AnalysisResult
from .ai.pipeline import run_pipeline, AnalysisError

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Upload validation
# ------------------------------------------------------------------
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
MAX_VIDEO_SIZE = 200 * 1024 * 1024  # 200 MB


def _validate_video(video_file):
    """Return an error string, or None if the file is acceptable."""
    if video_file is None:
        return "Please select a video to upload."

    name = getattr(video_file, "name", "") or ""
    ext = os.path.splitext(name)[1].lower()
    if ext not in ALLOWED_VIDEO_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_VIDEO_EXTENSIONS))
        return f"Unsupported file type '{ext or 'unknown'}'. Allowed: {allowed}."

    size = getattr(video_file, "size", 0) or 0
    if size <= 0:
        return "The uploaded file appears to be empty."
    if size > MAX_VIDEO_SIZE:
        return "The video is too large (max 200 MB)."

    return None


# ----------------------------
# Home Page
# ----------------------------
def home(request):
    return render(request, "home.html")


# ----------------------------
# Register Page
# ----------------------------
def register_page(request):

    if request.method == "POST":

        username = request.POST.get("username")
        email = request.POST.get("email")
        password = request.POST.get("password")

        if User.objects.filter(username=username).exists():
            messages.error(request, "Username already exists!")
            return redirect("register")

        User.objects.create_user(
            username=username,
            email=email,
            password=password
        )

        messages.success(request, "Registration Successful!")
        return redirect("login")

    return render(request, "register.html")


# ----------------------------
# Login Page
# ----------------------------
def login_page(request):

    if request.method == "POST":

        username = request.POST.get("username")
        password = request.POST.get("password")

        user = authenticate(
            request,
            username=username,
            password=password
        )

        if user is not None:
            login(request, user)
            return redirect("dashboard")

        messages.error(request, "Invalid Username or Password")

    return render(request, "login.html")


# ----------------------------
# Dashboard
# ----------------------------
@login_required(login_url="login")
def dashboard(request):

    analyses = AnalysisResult.objects.filter(
        user=request.user
    ).select_related("video").order_by("-uploaded_at")

    total_analyses = analyses.count()
    completed = analyses.filter(status="COMPLETED").count()

    best_score = None
    completed_scores = [
        a.overall_score for a in analyses if a.status == "COMPLETED"
    ]
    if completed_scores:
        best_score = max(completed_scores)

    recent = analyses[:6]

    context = {
        "analyses": analyses,
        "recent": recent,
        "total_analyses": total_analyses,
        "completed_analyses": completed,
        "best_score": best_score,
        "total_videos": UploadedVideo.objects.filter(user=request.user).count(),
    }
    return render(request, "dashboard.html", context)


# ----------------------------
# Logout
# ----------------------------
@login_required(login_url="login")
def logout_page(request):
    logout(request)
    return redirect("login")


# ----------------------------
# Upload Video + run AI pipeline
# ----------------------------
@login_required(login_url="login")
def upload_video(request):

    if request.method == "POST":

        title = (request.POST.get("title") or "").strip()
        if not title:
            title = "Untitled Video"

        video = request.FILES.get("video")

        error = _validate_video(video)
        if error:
            messages.error(request, error)
            return redirect("upload")

        saved = UploadedVideo.objects.create(
            user=request.user,
            title=title,
            video=video,
        )

        # Create the analysis row up front so status/errors are tracked
        result = AnalysisResult.objects.create(
            user=request.user,
            video=saved,
            title=title,
            status="PROCESSING",
        )

        # Resolve input + output paths
        video_path = saved.video.path
        out_dir = os.path.join(settings.MEDIA_ROOT, "videos", "processed")
        os.makedirs(out_dir, exist_ok=True)
        out_name = f"processed_{uuid.uuid4().hex}.mp4"
        out_path = os.path.join(out_dir, out_name)

        try:
            data = run_pipeline(video_path, output_path=out_path)

            result.total_frames = data["total_frames"]
            result.detected_frames = data["detected_frames"]
            result.detection_percentage = data["detection_percentage"]
            result.detected_shot = data["detected_shot"]
            result.shot_confidence = data["shot_confidence"]
            result.shot_reasoning = data["shot_reasoning"]

            scores = data["scores"]
            result.overall_score = scores["overall"]
            result.posture_score = scores["posture"]
            result.balance_score = scores["balance"]
            result.lower_body_score = scores["lower_body"]
            result.upper_body_score = scores["upper_body"]
            result.follow_through_score = scores["follow_through"]

            result.measurements = data["measurements"]
            result.coaching_feedback = data["feedback"]
            result.key_observations = data["observations"]

            result.processed_video.name = f"videos/processed/{out_name}"
            result.status = "COMPLETED"
            result.save()

            messages.success(
                request,
                f"Analysis complete — {result.detected_shot} "
                f"({result.overall_score}/100).",
            )
            return redirect("analysis_result", pk=result.pk)

        except AnalysisError as exc:
            result.status = "FAILED"
            result.error_message = str(exc)
            result.save()
            logger.warning("AI analysis failed for video %s: %s", saved.pk, exc)
            messages.error(request, f"Analysis could not be completed: {exc}")
            return redirect("analysis_result", pk=result.pk)

        except Exception as exc:  # noqa: BLE001 - never leak tracebacks to users
            result.status = "FAILED"
            result.error_message = "Unexpected processing error."
            result.save()
            logger.exception("Unexpected AI analysis failure for video %s", saved.pk)
            messages.error(
                request,
                "Analysis failed unexpectedly. "
                "Please try a different video.",
            )
            return redirect("analysis_result", pk=result.pk)

    return render(request, "upload.html")


# ----------------------------
# Analysis result page
# ----------------------------
@login_required(login_url="login")
def analysis_result(request, pk):
    # Ownership is enforced here: a user can only ever see their own
    # analysis (and therefore only their own processed video URL).
    result = get_object_or_404(
        AnalysisResult,
        pk=pk,
        user=request.user,
    )

    # Build display lists (kept out of the template for clarity)
    score_items = [
        ("Posture", result.posture_score),
        ("Balance", result.balance_score),
        ("Lower Body", result.lower_body_score),
        ("Upper Body", result.upper_body_score),
        ("Follow Through", result.follow_through_score),
    ]

    # feedback items are stored dicts; separate the disclaimer out
    feedback_items = result.coaching_feedback or []

    context = {
        "result": result,
        "score_items": score_items,
        "feedback_items": feedback_items,
    }
    return render(request, "result.html", context)

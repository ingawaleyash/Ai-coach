"""
End-to-end smoke test of the AI Coach Cricketer flow.

Uses Django's test client to simulate:
    register -> login -> upload video -> AI analysis -> view result
    -> dashboard history -> security (ownership isolation)

Run: python e2e_test.py
"""
import os
import sys
import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ai_coach.settings")
django.setup()

from django.test import Client  # noqa: E402
from django.urls import reverse  # noqa: E402
from django.conf import settings  # noqa: E402
from analysis.models import AnalysisResult, UploadedVideo  # noqa: E402

# Django's test client uses "testserver" as the host; allow it for this
# script only (does not touch the real settings file).
settings.ALLOWED_HOSTS = ["testserver", "localhost", "127.0.0.1"]

VIDEO_PATH = os.path.join("media", "videos", "flip.mp4")
USERNAME = "e2e_user"
PASSWORD = "TestPass_123"

passed = []
failed = []


def check(name, cond, extra=""):
    if cond:
        passed.append(name)
        print(f"  [PASS] {name}")
    else:
        failed.append(name)
        print(f"  [FAIL] {name} {extra}")


def main():
    c = Client()
    from django.contrib.auth.models import User
    User.objects.filter(username=USERNAME).delete()

    print("\n=== 1. REGISTER ===")
    r = c.post(reverse("register"), {
        "username": USERNAME,
        "email": "e2e@example.com",
        "password": PASSWORD,
    })
    check("register creates user", r.status_code == 302, r.status_code)

    print("\n=== 2. LOGIN ===")
    r = c.post(reverse("login"), {
        "username": USERNAME,
        "password": PASSWORD,
    })
    dest = getattr(r, "url", "")
    check("login redirects to dashboard", r.status_code == 302
          and reverse("dashboard") in dest, f"{r.status_code} {dest}")

    print("\n=== 3. LOGIN PAGE RENDERS (google button safety) ===")
    r = c.get(reverse("login"))
    check("login page renders 200", r.status_code == 200, r.status_code)

    print("\n=== 4. DASHBOARD ===")
    r = c.get(reverse("dashboard"))
    check("dashboard renders 200", r.status_code == 200, r.status_code)

    print("\n=== 5. UPLOAD + FULL AI ANALYSIS ===")
    with open(VIDEO_PATH, "rb") as f:
        r = c.post(reverse("upload"), {
            "title": "E2E Test Video",
            "video": f,
        })
    check("upload redirects", r.status_code in (302, 200), r.status_code)
    # Either redirect to result (success) or re-render with error
    results = AnalysisResult.objects.filter(user__username=USERNAME)
    check("analysis row created", results.exists())
    if results.exists():
        res = results.latest("id")
        check("status COMPLETED", res.status == "COMPLETED", res.status)
        check("detection > 0", res.detected_frames > 0,
              f"detected={res.detected_frames}")
        check("shot classified", bool(res.detected_shot), res.detected_shot)
        check("overall score in 0..100",
              0 <= res.overall_score <= 100, res.overall_score)
        check("processed video set", bool(res.processed_video),
              res.processed_video)
        check("feedback present", bool(res.coaching_feedback))
        check("observations present", bool(res.key_observations))

        print("\n=== 6. RESULT PAGE ===")
        r = c.get(reverse("analysis_result", args=[res.pk]))
        check("result page 200", r.status_code == 200, r.status_code)
        body = r.content.decode()
        check("page shows shot", res.detected_shot in body)
        check("page shows score", str(res.overall_score) in body)
        check("page shows video", b"<video" in r.content)
        print(f"      shot={res.detected_shot} "
              f"conf={res.shot_confidence:.2f} "
              f"overall={res.overall_score} "
              f"detection={res.detection_percentage:.1f}%")

        print("\n=== 7. SECURITY: OTHER USER CANNOT VIEW ===")
        other = User.objects.create_user("e2e_other", "o@e.com", PASSWORD)
        oc = Client()
        oc.login(username="e2e_other", password=PASSWORD)
        r = oc.get(reverse("analysis_result", args=[res.pk]))
        check("other user blocked (404)", r.status_code == 404, r.status_code)
        other.delete()

    print("\n=== 8. INVALID UPLOAD (no file) ===")
    r = c.post(reverse("upload"), {"title": "Bad"})
    check("no file -> redirect with error", r.status_code == 302, r.status_code)

    print("\n=== 8b. INVALID UPLOAD (bad extension) ===")
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tf:
        tf.write(b"not a video")
        tfname = tf.name
    with open(tfname, "rb") as f:
        r = c.post(reverse("upload"), {"title": "Bad", "video": f})
    os.unlink(tfname)
    check("bad ext -> redirect with error", r.status_code == 302, r.status_code)

    print("\n=== 9. UPLOAD PAGE 200 ===")
    r = c.get(reverse("upload"))
    check("upload page renders", r.status_code == 200, r.status_code)

    print("\n=== 10. HOME / REGISTER PAGES ===")
    check("home 200", c.get(reverse("home")).status_code == 200)
    check("register page 200", c.get(reverse("register")).status_code == 200)

    print("\n" + "=" * 50)
    print(f"PASSED: {len(passed)}   FAILED: {len(failed)}")
    if failed:
        print("FAILURES:", failed)
        sys.exit(1)
    print("ALL E2E CHECKS PASSED")


if __name__ == "__main__":
    main()

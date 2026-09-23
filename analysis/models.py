from django.db import models
from django.contrib.auth.models import User


class Player(models.Model):
    name = models.CharField(max_length=100)
    age = models.IntegerField()
    batting_style = models.CharField(max_length=50)

    def __str__(self):
        return self.name


class UploadedVideo(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    title = models.CharField(max_length=200)
    video = models.FileField(upload_to="videos/")
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title


class AnalysisResult(models.Model):

    # --- Processing status ---
    STATUS_CHOICES = [
        ("PENDING", "Pending"),
        ("PROCESSING", "Processing"),
        ("COMPLETED", "Completed"),
        ("FAILED", "Failed"),
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE)
    video = models.ForeignKey(
        UploadedVideo,
        on_delete=models.CASCADE,
        related_name="analyses",
    )
    title = models.CharField(max_length=200, blank=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default="PENDING",
    )

    # --- Processed output ---
    processed_video = models.FileField(
        upload_to="videos/processed/",
        blank=True,
        null=True,
    )

    # --- Frame statistics ---
    total_frames = models.IntegerField(default=0)
    detected_frames = models.IntegerField(default=0)
    detection_percentage = models.FloatField(default=0.0)

    # --- Shot classification ---
    detected_shot = models.CharField(max_length=100, blank=True, default="")
    shot_confidence = models.FloatField(default=0.0)

    # --- Technique scores (0-100) ---
    overall_score = models.IntegerField(default=0)
    posture_score = models.IntegerField(default=0)
    balance_score = models.IntegerField(default=0)
    lower_body_score = models.IntegerField(default=0)
    upper_body_score = models.IntegerField(default=0)
    follow_through_score = models.IntegerField(default=0)

    # --- Rich data ---
    measurements = models.JSONField(default=dict, blank=True)
    coaching_feedback = models.JSONField(default=list, blank=True)
    key_observations = models.JSONField(default=list, blank=True)
    shot_reasoning = models.JSONField(default=list, blank=True)

    # --- Errors ---
    error_message = models.TextField(blank=True, default="")

    def __str__(self):
        return f"{self.title or self.video.title} — {self.detected_shot} ({self.overall_score})"

    @property
    def measurements_display(self):
        """Curated, human-readable measurement rows for the results page."""
        summary = self.measurements.get("summary", {}) if self.measurements else {}
        labels = [
            ("l_elbow", "Left Elbow Angle", "°"),
            ("r_elbow", "Right Elbow Angle", "°"),
            ("l_knee", "Left Knee Angle", "°"),
            ("r_knee", "Right Knee Angle", "°"),
            ("l_hip", "Left Hip Angle", "°"),
            ("r_hip", "Right Hip Angle", "°"),
            ("torso_angle", "Torso Inclination", "°"),
            ("shoulder_slope", "Shoulder Alignment", "°"),
            ("hip_slope", "Hip Alignment", "°"),
            ("stance_width", "Stance Width", "× torso"),
            ("balance_offset", "Balance Offset", "× torso"),
            ("head_x_offset", "Head X-Offset", "× torso"),
        ]
        rows = []
        for key, label, unit in labels:
            stats = summary.get(key)
            if not stats:
                continue
            rows.append({
                "label": label,
                "unit": unit,
                "mean": stats.get("mean"),
                "min": stats.get("min"),
                "max": stats.get("max"),
                "range": stats.get("range"),
            })
        return rows
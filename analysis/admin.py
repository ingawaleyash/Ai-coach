from django.contrib import admin
from .models import Player, UploadedVideo, AnalysisResult


@admin.register(Player)
class PlayerAdmin(admin.ModelAdmin):
    list_display = ("name", "age", "batting_style")
    search_fields = ("name",)


@admin.register(UploadedVideo)
class UploadedVideoAdmin(admin.ModelAdmin):
    list_display = ("title", "user", "uploaded_at")
    list_filter = ("uploaded_at",)
    search_fields = ("title", "user__username")
    date_hierarchy = "uploaded_at"


@admin.register(AnalysisResult)
class AnalysisResultAdmin(admin.ModelAdmin):
    list_display = (
        "title",
        "user",
        "uploaded_at",
        "status",
        "detected_shot",
        "overall_score",
    )
    list_filter = ("status", "detected_shot", "uploaded_at")
    search_fields = ("title", "user__username", "detected_shot")
    date_hierarchy = "uploaded_at"
    readonly_fields = (
        "uploaded_at",
        "total_frames",
        "detected_frames",
        "detection_percentage",
        "detected_shot",
        "shot_confidence",
        "overall_score",
        "posture_score",
        "balance_score",
        "lower_body_score",
        "upper_body_score",
        "follow_through_score",
        "measurements",
        "coaching_feedback",
        "key_observations",
        "shot_reasoning",
        "processed_video",
    )

    def has_add_permission(self, request):
        # Results are created by the pipeline, not manually.
        return False

from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages

from .models import UploadedVideo


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
    return render(request, "dashboard.html")


# ----------------------------
# Logout
# ----------------------------
@login_required(login_url="login")
def logout_page(request):
    logout(request)
    return redirect("login")


# ----------------------------
# Upload Video
# ----------------------------
@login_required(login_url="login")
def upload_video(request):

    if request.method == "POST":

        title = request.POST.get("title")
        video = request.FILES.get("video")

        if not video:
            messages.error(request, "Please select a video.")
            return redirect("upload")

        UploadedVideo.objects.create(
            user=request.user,
            title=title,
            video=video
        )

        messages.success(request, "Video Uploaded Successfully!")
        return redirect("dashboard")

    return render(request, "upload.html")
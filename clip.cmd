@echo off
rem Usage: clip "path\to\video.mp4" [options]   (run "clip --help" for all options)
"%~dp0.venv\Scripts\python.exe" -m clipper %*

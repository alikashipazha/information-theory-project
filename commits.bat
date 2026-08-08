@echo off
echo Starting Git commit process...

:: 1. اضافه کردن فایل‌های پایه پروژه
git add README.md requirements.txt run.py
git commit -m "Initial commit: Add project structure, requirements and entry point"

:: 2. اضافه کردن لایه فیزیکی و بخش امنیت
git add src/physical_layer.py src/security.py
git commit -m "feat: Implement physical layer simulation and security/encryption module"

:: 3. اضافه کردن پروتکل ARQ و لایه انتقال داده
git add src/arq_protocol.py src/data_link_layer.py
git commit -m "feat: Implement ARQ protocol and data link layer logic"

:: 4. اضافه کردن رابط کاربری و تحلیل عملکرد
git add src/__init__.py src/gui.py src/performance_analysis.py src/main.py
git commit -m "feat: Add GUI, performance analysis module, and main application controller"

:: 5. اضافه کردن تست‌های شبیه‌ساز
git add tests/test_simulator.py
git commit -m "test: Add unit tests for simulator modules"

:: 6. اضافه کردن مستندات پروژه
git add docs/Project_requirements.pdf docs/Project_report.pdf
git commit -m "docs: Add project requirements and final report PDFs"

:: 7. آپلود فایل بات ایجاد شده (اختیاری)
git add commits.bat
git commit -m "chore: Add batch script for automated commits"

:: ارسال تغییرات به مخزن گیت‌هاب
echo Pushing changes to GitHub...
git push -u origin main

echo Process completed successfully.
pause
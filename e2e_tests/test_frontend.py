import pytest
from playwright.sync_api import Page
from pages.dashboard_page import DashboardPage

# Внутренний адрес в Docker сети
URL = "http://project-frontend"


def test_dashboard_functionality(page: Page):
    dashboard = DashboardPage(page)

    # 1. Открываем страницу
    print(f"Opening {URL}...")
    try:
        dashboard.navigate(URL)
    except:
        # Если вдруг упало по таймауту навигации, пробуем еще раз (бывает при высокой нагрузке)
        dashboard.navigate(URL)

    # 2. Проверяем, что загрузилась таблица данных
    print("Checking data loading...")
    dashboard.is_loaded()

    # 3. Проверяем, что сгенерировались графики
    print("Checking images...")
    dashboard.has_images()
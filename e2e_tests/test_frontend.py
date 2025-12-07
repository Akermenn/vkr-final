import pytest
from playwright.sync_api import Page
from pages.dashboard_page import DashboardPage

# Внутри Docker сети фронтенд доступен по имени контейнера
URL = "http://project-frontend"


def test_dashboard_functionality(page: Page):
    dashboard = DashboardPage(page)

    # 1. Открытие страницы
    print(f"Opening {URL}...")
    dashboard.navigate(URL)

    # 2. Проверка, что данные с бэкенда пришли
    dashboard.is_loaded()

    # 3. Проверка графиков
    dashboard.has_images()
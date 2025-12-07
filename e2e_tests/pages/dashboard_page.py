from playwright.sync_api import Page, expect

class DashboardPage:
    def __init__(self, page: Page):
        self.page = page
        self.metrics_card = page.locator("#metrics")
        self.corr_img = page.locator("#img-corr")

    def navigate(self, url):
        self.page.goto(url)
        # Ждем пока сеть успокоится (загрузятся картинки)
        self.page.wait_for_load_state("networkidle")

    def is_loaded(self):
        # Проверяем, что текст "Загрузка" исчез
        expect(self.metrics_card).not_to_contain_text("Загрузка данных")
        # Проверяем, что появилась точность
        expect(self.metrics_card).to_contain_text("Точность")

    def has_images(self):
        expect(self.corr_img).to_be_visible()
from playwright.sync_api import Page, expect


class DashboardPage:
    def __init__(self, page: Page):
        self.page = page
        # Локатор блока, куда загружается таблица
        self.results_block = page.locator("#class-results")

        # Локаторы картинок
        self.corr_img = page.locator("#img-corr")
        self.reg_img = page.locator("#img-reg")
        self.clust_img = page.locator("#img-clust")

    def navigate(self, url):
        self.page.goto(url)
        # Ждем загрузки сети
        self.page.wait_for_load_state("networkidle")

    def is_loaded(self):
        # 1. Проверяем, что надпись "Загрузка..." исчезла
        expect(self.results_block).not_to_contain_text("Загрузка данных")

        # 2. Проверяем, что появилась таблица с результатами
        # Ищем слово "Модель" (заголовок) или название одной из моделей
        expect(self.results_block).to_contain_text("Модель")
        expect(self.results_block).to_contain_text("Logistic Regression")

    def has_images(self):
        # Проверяем, что картинки видимы
        expect(self.corr_img).to_be_visible()
        expect(self.reg_img).to_be_visible()
        expect(self.clust_img).to_be_visible()
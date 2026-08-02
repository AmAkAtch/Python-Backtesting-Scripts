import os
from dotenv import load_dotenv
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

load_dotenv()

class TradingViewBot:
    PASSWORD = os.getenv("PASSWORD_TRADINGVIEW")
    EMAIL = os.getenv("EMAIL_TRADINGVIEW")

    def __init__(self,driver):
        self.driver = driver

    def open_and_login_trading_view(self):
        self.driver.get("https://www.tradingview.com/accounts/signin/")

        #find and click email button
        email_button = WebDriverWait(self.driver, 15).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "button[name='Email']"))
        )
        email_button.click()

        #Enter email and password
        input_email = WebDriverWait(self.driver,15).until(EC.presence_of_element_located((By.CSS_SELECTOR, "input[id='id_username']")))
        input_email.send_keys(self.EMAIL)
        input_password = WebDriverWait(self.driver,15).until(EC.presence_of_element_located((By.CSS_SELECTOR, "input[id='id_password']")))
        input_password.send_keys(self.PASSWORD, Keys.ENTER)

        #Check if captcha is present
        try:
            captcha_box = WebDriverWait(self.driver, 15).until(
                EC.presence_of_element_located((By.TAG_NAME, "iframe"))
            )
            input("Captcha found, solve it and than proceed....")
        except:
            print("No captcha is present, Proceeding...")

        

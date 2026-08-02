from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from trading_view_bot import TradingViewBot

options=Options()
options.add_experimental_option("detach", True)

driver = webdriver.Chrome(options=options)
trade_bot = TradingViewBot(driver)

trade_bot.open_and_login_trading_view()
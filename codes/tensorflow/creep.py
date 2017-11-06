# coding=utf-8

import os
from selenium import webdriver


class Spider:
    def __init__(self):
        self.page = 1
        self.dirName = 'ItemInformation'
        cap = webdriver.DesiredCapabilities.PHANTOMJS
        cap["phantomjs.page.settings.resourceTimeout"] = 1000
        # cap["phantomjs.page.settings.loadImages"] = False
        # cap["phantomjs.page.settings.localToRemoteUrlAccessEnabled"] = True
        self.driver = webdriver.PhantomJS(desired_capabilities=cap, service_log_path=os.path.devnull)

    def login(self):
        url = "http://10.108.255.249/"
        # load page
        self.driver.get(url)
        self.driver.switch_to_window(self.driver.current_window_handle)
        print(self.driver.current_url)

        # get dom object
        login_ = self.driver.find_element_by_id("loginname")
        password_ = self.driver.find_element_by_id("password")
        login_button = self.driver.find_element_by_id("button")

        print("make data")
        # set data
        login_.clear()
        # username  such as 143020100..
        login_.send_keys("14302010040")
        password_.clear()
        # password such as 123456
        password_.send_keys("940706dcjbxx")

        # start login
        print("start login")
        login_button.submit()

if __name__ == "__main__":
    spider = Spider()
    spider.login()

# from playwright.async_api import async_playwright
# import time
# import json
# import logging
# import os
# import asyncio

# # 确保日志目录存在
# log_dir = "logs"
# if not os.path.exists(log_dir):
#     os.makedirs(log_dir)

# # 创建一个专门的日志处理器
# file_handler = logging.FileHandler(os.path.join(log_dir, 'baidu_hot_news.log'),
#                                    encoding='utf-8')
# file_handler.setFormatter(
#     logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

# logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)
# logger.addHandler(file_handler)


# class BaiduHotNews:

#     def __init__(self):
#         logger.info("初始化 BaiduHotNews 实例")
#         self.browser = None
#         self.context = None
#         self.page = None
#         self.playwright = None

#     # 同步上下文管理器
#     def __enter__(self):
#         return self

#     def __exit__(self, exc_type, exc_val, exc_tb):
#         if self.browser:
#             self.browser.close()
#         if self.playwright:
#             self.playwright.stop()

#     # 异步上下文管理器
#     async def __aenter__(self):
#         return self

#     async def __aexit__(self, exc_type, exc_val, exc_tb):
#         if self.browser:
#             await self.browser.close()
#         if self.playwright:
#             await self.playwright.stop()

#     async def init_browser(self):
#         """初始化浏览器"""
#         try:
#             logger.info("开始初始化浏览器...")
#             self.playwright = await async_playwright().start()
#             logger.info("Playwright 启动成功")

#             self.browser = await self.playwright.chromium.launch(
#                 headless=True,
#                 args=[
#                     '--disable-gpu', '--disable-dev-shm-usage', '--no-sandbox',
#                     '--disable-setuid-sandbox'
#                 ])
#             logger.info("浏览器启动成功")

#             self.context = await self.browser.new_context(
#                 viewport={
#                     'width': 1920,
#                     'height': 1080
#                 },
#                 user_agent=
#                 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
#             self.page = await self.context.new_page()
#             logger.info("页面创建成功")

#         except Exception as e:
#             logger.error(f"浏览器初始化失败: {str(e)}", exc_info=True)
#             raise

#     async def scrape(self):
#         """抓取百度热搜原始数据
#         Returns:
#             list: 包含热搜信息的列表，格式为 [{"title": str, "link": str, "hot_index": int}]
#         """
#         try:
#             await self.init_browser()

#             # 访问百度热搜页面
#             await self.page.goto('https://top.baidu.com/board?tab=realtime')

#             # 等待页面加载完成
#             await self.page.wait_for_load_state('networkidle')
#             await self.page.wait_for_selector(
#                 'xpath=//div[contains(@class, "category-wrap_")]')

#             await asyncio.sleep(2)

#             # 获取热搜列表
#             hot_items = await self.page.query_selector_all(
#                 'xpath=//div[contains(@class, "category-wrap_")]')

#             hot_news_list = []

#             for item in hot_items:
#                 try:
#                     # 获取标题
#                     title_element = await item.query_selector(
#                         'xpath=.//div[contains(@class, "c-single-text-ellipsis")]'
#                     )
#                     if not title_element:
#                         continue

#                     title = await title_element.inner_text()
#                     title = title.strip()

#                     # 获取热搜指数并转换为整数
#                     index_element = await item.query_selector(
#                         'xpath=.//div[contains(@class, "hot-index_")]')
#                     hot_index = 0# 默认值
#                     if index_element:
#                         try:
#                             # 移除非数字字符并转换为整数
#                             hot_index_str = await index_element.inner_text()
#                             hot_index = int(''.join(
#                                 filter(str.isdigit, hot_index_str.strip())))
#                         except ValueError:
#                             hot_index = 0

#                     # 获取链接
#                     link_element = await item.query_selector(
#                         'xpath=.//a[contains(@class, "title_")]')
#                     link = await link_element.get_attribute(
#                         'href') if link_element else""

#                     if title:  # 只添加有标题的条目
#                         hot_news_list.append({
#                             'title': title,
#                             'link': link,
#                             'hot_index': hot_index
#                         })
#                         logger.info(f"成功解析: {title}")

#                 except Exception as e:
#                     logger.error(f"解析条目出错: {str(e)}")
#                     continue

#             return hot_news_list

#         except Exception as e:
#             logger.error(f"抓取过程出错: {str(e)}")
#             return []

#         finally:
#             if self.browser:
#                 await self.browser.close()

#     async def get_hot_news(self):
#         """获取百度热搜
#         Returns:
#             str: JSON字符串，包含百度热搜列表和错误信息
#         """
#         try:
#             results = await self.scrape()

#             if not results:
#                 return json.dumps({
#                     "error": "未获取到热搜数据",
#                     "data": []
#                 },
#                                   ensure_ascii=False)

#             return json.dumps({
#                 "error": None,
#                 "data": results
#             },
#                               ensure_ascii=False)

#         except Exception as e:
#             return json.dumps({
#                 "error": f"获取热搜失败: {str(e)}",
#                 "data": []
#             },
#                               ensure_ascii=False)


# if __name__ == '__main__':
#     # 使用示例
#     async def main():
#         async with BaiduHotNews() as scraper:
#             result = await scraper.get_hot_news()
#             data = json.loads(result)

#             if data["error"]:
#                 print(f"错误: {data['error']}")
#             else:
#                 print("\n今日热搜：")
#                 for idx, item in enumerate(data["data"], 1):
#                     print(f"\n{idx}. {item['title']}")
#                     print(f"热搜指数: {item['hot_index']}")
#                     print(f"链接: {item['link']}")

#     asyncio.run(main())


from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain_community.tools.playwright.utils import create_sync_playwright_browser
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.chat_models import init_chat_model

# 初始化 Playwright 浏览器：
sync_browser = create_sync_playwright_browser()
toolkit = PlayWrightBrowserToolkit.from_browser(sync_browser=sync_browser)
tools = toolkit.get_tools()

# 初始化大模型
model = init_chat_model(
    model="Qwen/Qwen3-8B",
    model_provider="openai",
    base_url="https://api.siliconflow.cn/v1",
    api_key="sk-hcebbggeoucrpvxhtakviksgvfwpqpmixzjycqqfpahgcfqu",
)

# 通过 Langchain Hub 拉取提示词，以自定义提示词等价
prompt = hub.pull("hwchase17/openai-tools-agent")

# 通过 Langchain 创建 OpenaAI 工具代理
agent = create_openai_tools_agent(model, tools, prompt)

# 通过 AgentExecutor 执行代理
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

if __name__ == "__main__":
    # 定义任务
    command = {
        "input": "访问这个网站 https://www.microsoft.com/en-us/microsoft-365/blog/2025/01/16/copilot-is-now-included-in-microsoft-365-personal-and-family/?culture=zh-cn&country=cn 并帮我总结一下这个网站的内容"
    }

    # 执行任务
    response = agent_executor.invoke(command)
    print(response)
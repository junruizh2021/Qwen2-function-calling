import json
import os

import json5
import urllib.parse
from qwen_agent.agents import Assistant
from qwen_agent.tools.base import BaseTool, register_tool

llm_cfg = {
    # Use the model service provided by DashScope:
    #'model': 'qwen-max',
    #'model_server': 'dashscope',
    # 'api_key': 'YOUR_DASHSCOPE_API_KEY',
    # It will use the `DASHSCOPE_API_KEY' environment variable if 'api_key' is not set here.

    # Use your own model service compatible with OpenAI API:
    'model': 'Qwen2-7B-Instruct',
    'model_server': 'http://10.239.152.95:8000/v1',  # api_base
    'api_key': 'EMPTY',

    # (Optional) LLM hyperparameters for generation:
    'generate_cfg': {
        'top_p': 0.8
    }
}
#system = 'According to the user\'s request, you first draw a picture and then automatically run code to download the picture ' + \
#          'and select an image operation from the given document to process the image'
system = 'You are a helpful assistant, and you can use some important tools to execute my command, including: my_image_gen, get_current_weather.'
# Add a custom tool named my_image_gen：
@register_tool('my_image_gen')
class MyImageGen(BaseTool):
    description = 'AI painting (image generation) service, input text description, and return the image URL drawn based on text information.'
    parameters = [{
        'name': 'prompt',
        'type': 'string',
        'description': 'Detailed description of the desired image content, in English',
        'required': True
    }]

    def call(self, params: str, **kwargs) -> str:
        prompt = json5.loads(params)['prompt']
        prompt = urllib.parse.quote(prompt)
        return json.dumps(
            {'image_url': f'https://image.pollinations.ai/prompt/{prompt}'},
            ensure_ascii=False)


@register_tool('get_current_weather')
class MyCurrentWeather(BaseTool):
    description = 'search weather of a given city in China and response the temperature'
    parameters = [{
        'name': 'location',
        'type': 'string',
        'description': 'The city of China, in chinese',
        'required': True
    },
    {
        'name': 'unit',
        'type': 'string',
        'enum': ['celsius', 'fahrenheit']}
    ]

    def call(self, params: str, **kwargs) -> str:
        print("params: ", params)
        params = json5.loads(params)
        location = params['location']
        unit = params['unit']
        if not isinstance(location, str):
            raise TypeError("City name must be a string")
        api_key = "Sg3PPWFJS6prWTT7x"
        url = f"https://api.seniverse.com/v3/weather/now.json?key={api_key}&location={location}&language=zh-Hans&unit=c"
        import requests
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return json.dumps({
                'location': location,
                'temperature': data["results"][0]["now"]["temperature"],
                'unit': unit
                })
        else:
            raise Exception(f"Failed to retrieve weather: {response.status_code}")

tools = ['my_image_gen', 'get_current_weather','code_interpreter']  # code_interpreter is a built-in tool in Qwen-Agent
bot = Assistant(llm=llm_cfg,
                system_message=system,
                function_list=tools,
                files=[os.path.abspath('doc.pdf')])

messages = []
while True:
    messages.append({'role': 'user', 'content': input('Input < ')})
    response = []
    for response in bot.run(messages=messages):
        print('bot response:', response)
    messages.extend(response)

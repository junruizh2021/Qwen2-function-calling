import json
import os
import json5
import torch
from bigdl.llm.transformers import AutoModelForCausalLM
from transformers import AutoTokenizer, TextStreamer
from transformers import TextIteratorStreamer
from qwen_generation_utils import StopWordsLogitsProcessor
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation import GenerationConfig
import urllib.parse
import requests
from threading import Thread
import re
import time
model_path = ("/home/anna/WorkSpace/celadon/demo-src/celadon-aigc/models/LLM/Qwen2-7B-Instruct-"
              "int4")
tokenizer = AutoTokenizer.from_pretrained(model_path)
generation_config = GenerationConfig.from_pretrained(model_path)
model = AutoModelForCausalLM.load_low_bit(model_path,
                                    trust_remote_code=True,
                                    optimize_model=False).eval()
model.generation_config = generation_config
model.generation_config.top_k = 1
model.to('xpu:0')

TOOL_DESC = """{name_for_model}: Call this tool to interact with the {name_for_human} API. What is the {name_for_human} API useful for? {description_for_model} Parameters: {parameters}"""

PROMPT_REACT = """Answer the following questions as best you can. You have access to the following APIs:

{tools_text}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tools_name_text}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question.

Begin!

Question: {query}"""


def llm_with_plugin(prompt: str, history, list_of_plugin_info=()):
    chat_history = [(x['user'], x['bot']) for x in history] + [(prompt, '')]

    planning_prompt = build_input_text(chat_history, list_of_plugin_info)
    text = ''
    count = 1
    new_history = []
    new_history.extend(history)
    stop_words=['Observation:', 'Observation:\n', 'answer the question.']
    while True:
        #print(f'\nchat_with_llm {count} epoch :\n')
        new_text = ''
        # should return the final_text for main to streaming-print
        for finalanswer, output in text_completion(planning_prompt + text, stop_words=stop_words):
            if finalanswer:
                #print('llm_plugin output : ', output)
                yield output
            else:
                #print('llm_plugin output : ', output)
                new_text += output
        new_text = new_text[len(planning_prompt + text) :]
        new_text=new_text.replace('<|endoftext|>', '').replace('<|im_end|>', '')
        for stop_str in stop_words:
            idx = new_text.find(stop_str)
            if idx != -1:
                new_text = new_text[: idx + len(stop_str)]
        #print("text is : ", new_text)
        #print('\nnew_text : ', new_text)
        action, action_input, output = parse_latest_plugin_call(new_text)
        #print("\naction : ", action)
        if action:
            observation = call_plugin(action, action_input)
            #print('\nobservation : \n', observation)
            output += f'\nObservation: {observation}\nThought:'
            text += output
        else:
            text += output
            break
        count += 1

def build_input_text(chat_history, list_of_plugin_info) -> str:
    tools_text = []
    for plugin_info in list_of_plugin_info:
        tool = TOOL_DESC.format(
            name_for_model=plugin_info["name_for_model"],
            name_for_human=plugin_info["name_for_human"],
            description_for_model=plugin_info["description_for_model"],
            parameters=json.dumps(plugin_info["parameters"], ensure_ascii=False),
        )
        if plugin_info.get('args_format', 'json') == 'json':
            tool += " Format the arguments as a JSON object."
        elif plugin_info['args_format'] == 'code':
            tool += ' Enclose the code within triple backticks (`) at the beginning and end of the code.'
        #else:
            #print('plugin_info', plugin_info['args_format'])
            #raise NotImplementedError
        tools_text.append(tool)
    tools_text = '\n\n'.join(tools_text)

    tools_name_text = ', '.join([plugin_info["name_for_model"] for plugin_info in list_of_plugin_info])

    im_start = '<|im_start|>'
    im_end = '<|im_end|>'
    prompt = f'{im_start}system\nYou are a helpful assistant. And you must response user questions according to the given format.{im_end}'
    for i, (query, response) in enumerate(chat_history):
        if list_of_plugin_info:
            if (len(chat_history) == 1) or (i == len(chat_history) - 2):
                query = PROMPT_REACT.format(
                    tools_text=tools_text,
                    tools_name_text=tools_name_text,
                    query=query,
                )
        query = query.lstrip('\n').rstrip()
        response = response.lstrip('\n').rstrip()
        prompt += f"\n{im_start}user\n{query}{im_end}"
        prompt += f"\n{im_start}assistant\n{response}{im_end}"

    assert prompt.endswith(f"\n{im_start}assistant\n{im_end}")
    prompt = prompt[: -len(f'{im_end}')]
    #print('prompt : ', prompt)
    return prompt


def text_completion(input_text, **kwargs):
    kwargs.setdefault('max_new_tokens', 2048)
    stop_words = kwargs.pop("stop_words", None)
    im_end = '<|im_end|>'
    if im_end not in stop_words:
        stop_words = stop_words + [im_end]
    stop_words_ids = [tokenizer.encode(w) for w in stop_words]
    #print(f"stop_words : {stop_words}")

    stop_words_logits_processor = StopWordsLogitsProcessor(
            stop_words_ids=stop_words_ids,
            eos_token_id=model.generation_config.eos_token_id,
            )
    logits_processor = LogitsProcessorList([stop_words_logits_processor])
    
    #text = tokenizer.apply_chat_template(input_text,tokenize=False,add_generation_prompt=False)
    
    #streamer = TextStreamer(tokenizer, skip_prompt=True)
    streamer = TextIteratorStreamer(tokenizer)
    
    input_ids = torch.tensor([tokenizer.encode(input_text)]).to('xpu:0')
    
    #output = model.generate(input_ids, logits_processor=logits_processor, **kwargs)
    generation_kwargs = dict(input_ids=input_ids, streamer=streamer, logits_processor=logits_processor, **kwargs)
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    output_history = ""
    finalanswer = False
    final_answer_text = ""
    count = 1
    Thought = True
    for output in streamer:
        #print(f'\n{count} epoch of streamer:\n')
        #if count>1:
        output_history += output
        #output_histrory = output_history[len(input_text):]
        #print("output_history :\n", output_history)
        #print("output now : ", output)
        # Don't change the endswith string 'Final Answer: '
        #print("\noutput now : \n", output)
        #if output_history.endswith('<|im_start|>assistant'):
        #    condition_1=True
        #if output_history.endswith('Thought: '):
        #    condition_1=False
        ## output don't use the template and answer the question directly
        #print('Thought : ', Thought, '\nfinalanswer : ', finalanswer)
        #if count==4 and 'Thought: ' not in output:
            #print('output print once: ', output)
        #    print(f'now is {count} epoch!')
            #print('Thought changed!')
        #    Thought = False
        #print('Thought : ', Thought, '\nfinalanswer : ', finalanswer)
        if output_history.endswith('Final Answer: ') or finalanswer or not Thought:
            final_answer_text += output
            finalanswer = True
        elif output.startswith('Answer: '):
            #output = output.split(":", 1)[-1]
            final_answer_text += output
            finalanswer = True
        #print('final_answer_text : ', final_answer_text, '\nThought : ', Thought, '\nfinalanswer : ', finalanswer)
        count+=1
        yield finalanswer, output
        #elif: 
        #    yield finalanswer, output
    ##output = output_history
    #output = output.tolist()[0]
    #output = tokenizer.decode(output, errors="ignore")
    #for token in streamer:
    #    generated_text += token
    #    print(generated_text, flush=True)
    #assert output.startswith(input_text)
    ##output = output.replace('<|endoftext|>', '').replace(im_end, '')
    #print("output : ", output)
    ##for stop_str in stop_words:
    ##    idx = output.find(stop_str)
    ##    if idx != -1:
    ##        output = output[: idx + len(stop_str)]

def llm_postprocess(input_text, **kwargs):
    kwargs.setdefault('max_new_tokens', 1024)
    input_ids = torch.tensor([tokenizer.encode(input_text)]).to('xpu:0')
    print('history token : ', input_ids.size(1))

    generation_kwargs = dict(input_ids, streamer=streamer, **kwargs)
    streamer = TextIteratorStreamer(tokenizer)
    thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
    thread.start()
    for output in streamer:
        yield token


def parse_latest_plugin_call(text):
    plugin_name, plugin_args = '', ''
    i = text.rfind('\nAction:')
    j = text.rfind('\nAction Input:')
    k = text.rfind('\nObservation:')
    if 0 <= i < j:  # If the text has `Action` and `Action input`,
        if k < j:  # but does not contain `Observation`,
            # then it is likely that `Observation` is ommited by the LLM,
            # because the output text may have discarded the stop word.
            text = text.rstrip() + '\nObservation:'  # Add it back.
        k = text.rfind('\nObservation:')
        plugin_name = text[i + len('\nAction:') : j].strip()
        plugin_args = text[j + len('\nAction Input:') : k].strip()
        text = text[:k]
    return plugin_name, plugin_args, text

def extract_final_answer(text):
    # 使用正则表达式匹配 'Final Answer: ' 后面的所有字符，直到字符串结尾
    match = re.search(r'Final Answer: (.*)', text, re.DOTALL)
    if match:
        return match.group(1).strip()  # strip() 去除首尾的空格
    return None

def call_plugin(plugin_name: str, plugin_args: str) -> str:
    if plugin_name == 'image_gen':
        prompt = json5.loads(plugin_args)["prompt"]
        prompt = urllib.parse.quote(prompt)
        return json.dumps({'image_url': f'https://image.pollinations.ai/prompt/{prompt}'}, ensure_ascii=False)
    elif plugin_name == 'get_current_weather':
        city = json5.loads(plugin_args)["city"]
        city = urllib.parse.quote(city)
        if not isinstance(city, str):
            raise TypeError("City name must be a string")

        api_key = "Sg3PPWFJS6prWTT7x"
        url = f"https://api.seniverse.com/v3/weather/now.json?key={api_key}&location={city}&language=zh-Hans&unit=c"
        response = requests.get(url)
        data = response.json()
        if response.status_code == 200:
            return json.dumps({'"temperature"': data["results"][0]["now"]["temperature"], "description": data["results"][0]["now"]["text"],}, ensure_ascii=False)
        else:
            raise Exception(f"Failed to retrieve weather: {response.status_code}")
    elif lugin_name == 'get_lunar':
        from datetime import date, datetime
        from lunar_python import Lunar, Solar

        solar_date = Solar.fromDate(datetime.now())
        lunar_date = Lunar.fromDate(datetime.now())

        festivals = ""
        for festival in solar_date.getFestivals():
            festivals += festival
            festivals += "，"
        for festival in solar_date.getOtherFestivals():
            festivals += festival
            festivals += "，"
        festivals = festivals[:-1]

        result = "solar date {}年{}月{}日星期{}，lunar date 农历{}年{}月{}, {}".format(solar_date.getYear(), solar_date.getMonth(), solar_date.getDay(), solar_date.getWeekInChinese(),
                                                                    lunar_date.getYearInGanZhi(), lunar_date.getMonthInChinese(), lunar_date.getDayInChinese(), festivals)

        return result
    elif lugin_name == 'car_controll':
        response = "车辆控制已经{}".format(plugin_args)
    else:
        raise NotImplementedError


def test():
    tools = [
        {
            'name_for_human': '文生图',
            'name_for_model': 'image_gen',
            'description_for_model': '文生图是一个AI绘画（图像生成）服务，输入文本描述，返回根据文本作画得到的图片的URL',
            'parameters': [
                {
                    'name': 'prompt',
                    'description': '英文关键词，描述了希望图像具有什么内容',
                    'required': True,
                    'schema': {'type': 'string'},
                }
            ],
        },
        {
            'name_for_human': '天气查询',
            'name_for_model': 'get_current_weather',
            'description_for_model': '天气查询工具通过调用天气API，获取给定城市的实时天气',
            'parameters': [
                {
                    'name': 'city',
                    'description': 'A city, in chinese',
                    'required': True,
                    'schema': {'type': 'string'},
                }
            ],
        },
        {
            'name_for_human': '查询农历日期',
            'name_for_model': 'get_lunar',
            'description_for_model': '使用当前日期和时间，获取对应的公历和农历日期',
            'parameters': [
                {}
            ],
        },
        {
            'name_for_human': '车辆控制',
            'name_for_model': 'car_control',
            'description_for_model': '用于控制车辆组件',
            'parameters': [
                {
                    'name': 'component',
                    'description': 'the component need to be controlled',
                    'required': True,
                    'schema': {'type': 'string'},
                },
                {
                    'name': 'command',
                    'description': 'how to control the component',
                    'required': True,
                    'schema': {'type': 'string'},
                }
            ],
        },
    ]
    history = []
    while True:
        #print('history :', history)
        query = input('\nInput < ')
        new_history = []
        new_history.extend(history)
        text = ''
        response_final = ''
        for response in llm_with_plugin(prompt=query, history=history, list_of_plugin_info=tools):
            if 'Answer:' in response:
                response_final = response.split(': ',1)[-1]
            elif '<|im_end|>' in response:
                response_final = response.split('<|im_end|>',1)[0]
            else:
                response_final = response
            text += response_final
            print(response_final, end='', flush=True)
        #new_history.append({'user': query, 'bot': text})
        history=new_history

        #print('\nresponse : ', response)
        #print(extract_final_answer(response))
        #for token in list(extract_final_answer(response)):
        #    time.sleep(0.02)
        #    print(token, end='', flush=True)
        #output = llm_postprocess()
        #for token in output_text:
        #    print(token)


if __name__ == "__main__":
    test()

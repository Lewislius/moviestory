import os
import sys
import json
import argparse
from pathlib import Path
from transformers import AutoModelForImageTextToText, AutoProcessor
from datetime import datetime

# 配置参数
VIDEO_INPUT_DIR = r"/home/liuzhirui/video_demo"  # 输入视频文件夹
OUTPUT_DIR = r"/home/liuzhirui/model/Qwen3-VL/captions"  # 输出文件夹
VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv']
MODEL_NAME = "Qwen/Qwen3-VL-4B-Instruct"

# 提示词配置 - 可以根据需要修改
# 单个提示词模式：
# PROMPT_TEXT = "Describe this video, put its main character first in your caption, and focus on describing the character's actions and expressions."

# 多个提示词模式（推荐）- 运行一次自动生成多个不同角度的描述：
PROMPT_TEXT = [
    # "Describe this video.",
    # "Describe this video, be brief.",
    "Describe this video, be brief, focusing on the characters and their actions. Note: Please provide a description of the characters, their actions, and any notable expressions or emotions they display. No new lines. Keep it under 30 words if possible.",
    "Describe this video, be brief, focusing on the characters and their actions. Note: Please provide a description of the characters, their actions, and any notable expressions or emotions they display. No new lines. Keep it under 60 words if possible.",
    # "Describe this video, be brief, focusing on the characters, their actions and demeanor.",
    # "Describe this video, be brief, focusing on the scene, the characters, their actions and demeanor.",
    "Describe this video, focusing on the characters and their actions. Note: Please provide a detailed description of the characters, their actions, and any notable expressions or emotions they display. No new lines. Keep it under 60 words if possible.",
    "Describe this video, focusing on the characters and their actions. Note: Please provide a detailed description of the characters, their actions, and any notable expressions or emotions they display. No new lines. Keep it under 100 words if possible.",
    # "Describe this video, focusing on the characters, their actions and demeanor.",
    # "Describe this video, focusing on the scene, the characters, their actions and demeanor.",
    # "Describe this video in detail, focusing on the characters and their actions.",
    # "Describe this video in detail, focusing on the characters, their actions and demeanor.",
    # "Describe this video in detail, including the scene, characters, and actions.",
    # "Describe this video in detail, focusing on the scene, the characters, their actions and demeanor.",
    # "Describe this video, emphasize the character’s actions and demeanor, and connect the character’s name to their actions.",
    # "Describe this video, put its main character first in your caption, and focus on describing the character's actions and expressions.",
]

# 其他可选提示词：
# PROMPT_TEXT = "Describe this video, with the main subject in the video appearing at the very beginning of the description."
# PROMPT_TEXT = ["Describe the main action", "Describe the emotions", "Describe the scene"]

def load_model(model_name=None):
    """加载模型和处理器"""
    if model_name is None:
        model_name = MODEL_NAME
    print(f"正在加载模型 {model_name} ...")
    model = AutoModelForImageTextToText.from_pretrained(
        model_name, dtype="auto", device_map="auto"
    )
    
    # We recommend enabling flash_attention_2 for better acceleration and memory saving
    # model = AutoModelForImageTextToText.from_pretrained(
    #     model_name,
    #     dtype=torch.bfloat16,
    #     attn_implementation="flash_attention_2",
    #     device_map="auto",
    # )
    
    processor = AutoProcessor.from_pretrained(model_name)
    print("模型加载完成！")
    return model, processor

def generate_caption(model, processor, video_path, prompt_text, ref_image_path=None):
    """为单个视频生成描述，可选附带角色参考图"""
    content = []
    # 如果有参考图，先插入参考图
    if ref_image_path:
        content.append({"type": "image", "image": ref_image_path})
    content.append({"type": "video", "video": video_path})
    content.append({"type": "text", "text": prompt_text})

    messages = [{"role": "user", "content": content}]
    
    # Preparation for inference
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)
    
    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=256)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    return output_text[0] if output_text else ""

def save_caption_result(video_path, prompt, answer, output_dir):
    """保存描述结果到文件（追加模式）"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取视频文件名（不含扩展名）
    video_name = Path(video_path).stem
    
    # 当前结果
    current_result = {
        "video_path": str(video_path),
        "video_name": video_name,
        "prompt": prompt,
        "caption": answer,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # JSON文件处理 - 以数组形式保存多个结果
    json_path = os.path.join(output_dir, f"{video_name}_caption.json")
    
    # 如果文件已存在，读取现有内容
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                # 如果现有数据不是列表，转换为列表
                if not isinstance(existing_data, list):
                    existing_data = [existing_data]
        except:
            existing_data = []
    else:
        existing_data = []
    
    # 追加新结果
    existing_data.append(current_result)
    
    # 保存更新后的数据
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=2)
    
    # TXT文件处理 - 追加模式
    txt_path = os.path.join(output_dir, f"{video_name}_caption.txt")
    
    # 使用追加模式打开文件
    with open(txt_path, 'a', encoding='utf-8') as f:
        # 如果文件已有内容，添加分隔符
        if os.path.exists(txt_path) and os.path.getsize(txt_path) > 0:
            f.write("\n" + "="*80 + "\n\n")
        
        f.write(f"视频名称: {video_name}\n")
        f.write(f"视频路径: {video_path}\n")
        f.write(f"提问内容: {prompt}\n")
        f.write(f"生成描述: {answer}\n")
        f.write(f"生成时间: {current_result['timestamp']}\n")
    
    print(f"✓ 结果已追加保存: {json_path}")
    return json_path

def find_videos(directory):
    """递归查找目录下的所有视频文件"""
    video_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in VIDEO_EXTENSIONS):
                video_files.append(os.path.join(root, file))
    return video_files

def process_single_video(video_path, output_dir=OUTPUT_DIR, prompt=PROMPT_TEXT):
    """处理单个视频"""
    model, processor = load_model()
    
    # 如果 prompt 是列表，遍历所有 prompt
    if isinstance(prompt, list):
        prompts = prompt
    else:
        prompts = [prompt]
    
    for idx, single_prompt in enumerate(prompts, 1):
        if len(prompts) > 1:
            print(f"\n[Prompt {idx}/{len(prompts)}] 正在处理视频: {video_path}")
            print(f"当前提示词: {single_prompt[:80]}...")  # 显示前80个字符
        else:
            print(f"\n正在处理视频: {video_path}")
        
        caption = generate_caption(model, processor, video_path, single_prompt)
        print(f"生成的描述: {caption}")
        
        save_caption_result(video_path, single_prompt, caption, output_dir)
    
    return caption

def process_videos_batch(input_dir=VIDEO_INPUT_DIR, output_dir=OUTPUT_DIR, prompt=PROMPT_TEXT):
    """批量处理视频文件夹"""
    # 查找所有视频文件
    video_files = find_videos(input_dir)
    
    if not video_files:
        print(f"在 {input_dir} 中没有找到视频文件")
        return
    
    print(f"找到 {len(video_files)} 个视频文件")
    
    # 检查 prompt 类型
    if isinstance(prompt, list):
        prompts = prompt
        print(f"将使用 {len(prompts)} 个不同的提示词对每个视频进行描述")
    else:
        prompts = [prompt]
    
    # 加载模型（只加载一次）
    model, processor = load_model()
    
    # 处理每个视频
    for video_idx, video_path in enumerate(video_files, 1):
        print(f"\n{'='*80}")
        print(f"[视频 {video_idx}/{len(video_files)}] {video_path}")
        print(f"{'='*80}")
        
        # 对当前视频使用所有 prompt
        for prompt_idx, single_prompt in enumerate(prompts, 1):
            try:
                if len(prompts) > 1:
                    print(f"\n  [Prompt {prompt_idx}/{len(prompts)}]")
                    print(f"  提示词: {single_prompt[:80]}...")
                
                caption = generate_caption(model, processor, video_path, single_prompt)
                print(f"  生成的描述: {caption}")
                
                save_caption_result(video_path, single_prompt, caption, output_dir)
                
            except Exception as e:
                print(f"  ✗ 处理失败: {e}")
                continue
    
    print(f"\n{'='*80}")
    print(f"✓ 批量处理完成！")
    print(f"  - 处理了 {len(video_files)} 个视频")
    print(f"  - 每个视频使用了 {len(prompts)} 个提示词")
    print(f"  - 总共生成了 {len(video_files) * len(prompts)} 个描述")
    print(f"  - 结果保存在: {output_dir}")
    print(f"{'='*80}")

# ====================== 单条处理模式（供 shell 脚本逐条调用） ======================

def process_single_with_ref(
    video_path: str,
    ref_image_path: str,
    output_json: str,
    prompts=PROMPT_TEXT,
    metadata_json: str = None,
    model_name: str = None,
):
    """
    处理单个 segment + 参考图，将结果追加到 output_json。
    供 batch_caption.sh 逐条调用。
    """
    if isinstance(prompts, str):
        prompts = [prompts]

    model, processor = load_model(model_name)

    # 读取 metadata（如果提供）
    char_metadata = None
    if metadata_json and os.path.exists(metadata_json):
        try:
            with open(metadata_json, 'r', encoding='utf-8') as f:
                char_metadata = json.load(f)
        except Exception:
            pass

    # 读取已有结果（追加模式）
    existing = _load_output_json(output_json)

    for idx, prompt in enumerate(prompts, 1):
        print(f"  [Prompt {idx}/{len(prompts)}] {prompt[:80]}...")
        try:
            caption = generate_caption(model, processor, video_path, prompt, ref_image_path)
            print(f"  → {caption[:120]}")
            existing.append({
                "segment_video_path": video_path,
                "character_crop_path": ref_image_path,
                "video_path": char_metadata.get("video_path") if char_metadata else None,
                "bbox": char_metadata.get("bbox") if char_metadata else None,
                "prompt": prompt,
                "caption": caption,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            })
        except Exception as e:
            print(f"  ✗ 生成失败: {e}")

    _save_output_json(output_json, existing)
    print(f"  ✓ 结果已保存: {output_json}")


# ====================== 任务清单批量模式（供 shell 脚本调用，模型只加载一次） ======================

def _load_output_json(path: str) -> list:
    """读取已有的输出 JSON，失败时返回空列表。"""
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data if isinstance(data, list) else [data]
        except Exception:
            pass
    return []


def _save_output_json(path: str, data: list):
    """覆盖写入输出 JSON。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def process_batch_ref(tasklist_json: str, prompts=PROMPT_TEXT, model_name: str = None, skip_existing: bool = True):
    """
    读取 shell 脚本生成的任务清单 JSON，加载一次模型，批量处理所有任务。

    任务清单格式（每条）：
    {
      "segment_video": "<绝对路径>.mp4",
      "crop_image":    "<绝对路径>_crop.jpg",
      "metadata_json": "<绝对路径>_metadata.json" | null,
      "output_json":   "<剧集文件夹>/captions_output.json"
    }
    """
    if isinstance(prompts, str):
        prompts = [prompts]

    # 读取任务清单
    with open(tasklist_json, 'r', encoding='utf-8') as f:
        tasks = json.load(f)

    if not tasks:
        print("任务清单为空，退出。")
        return

    print(f"共读取 {len(tasks)} 条任务，每条使用 {len(prompts)} 个提示词")
    print(f"总计推理次数: {len(tasks) * len(prompts)}")

    # 加载模型（只加载一次）
    model, processor = load_model(model_name)

    # 按 output_json 分组缓存已有结果，避免每次都读写磁盘
    output_cache: dict[str, list] = {}   # output_json_path -> list of results
    skip_keys:    dict[str, set]  = {}   # output_json_path -> set of done keys

    def _get_cache(out_path: str):
        if out_path not in output_cache:
            existing = _load_output_json(out_path)
            output_cache[out_path] = existing
            if skip_existing:
                skip_keys[out_path] = {
                    f"{r.get('segment_video_path','')}|{r.get('character_crop_path','')}|{r.get('prompt','')}"
                    for r in existing
                }
            else:
                skip_keys[out_path] = set()
        return output_cache[out_path], skip_keys[out_path]

    done = skipped = failed = 0
    total = len(tasks) * len(prompts)

    for task_idx, task in enumerate(tasks, 1):
        seg_video  = task["segment_video"]
        crop_image = task["crop_image"]
        out_json   = task["output_json"]
        meta_path  = task.get("metadata_json")

        # 读取 metadata
        char_metadata = None
        if meta_path and os.path.exists(meta_path):
            try:
                with open(meta_path, 'r', encoding='utf-8') as f:
                    char_metadata = json.load(f)
            except Exception:
                pass

        results, done_keys = _get_cache(out_json)

        print(f"\n{'='*70}")
        print(f"[{task_idx}/{len(tasks)}] {Path(seg_video).name}  |  {Path(crop_image).name}")

        for p_idx, prompt in enumerate(prompts, 1):
            key = f"{seg_video}|{crop_image}|{prompt}"

            if key in done_keys:
                print(f"  [Prompt {p_idx}] 已存在，跳过")
                skipped += 1
                done += 1
                continue

            print(f"  [Prompt {p_idx}/{len(prompts)}] {prompt[:80]}...")
            try:
                caption = generate_caption(model, processor, seg_video, prompt, crop_image)
                print(f"  → {caption[:120]}")

                entry = {
                    "segment_video_path": seg_video,
                    "character_crop_path": crop_image,
                    "video_path": char_metadata.get("video_path") if char_metadata else None,
                    "bbox": char_metadata.get("bbox") if char_metadata else None,
                    "prompt": prompt,
                    "caption": caption,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
                results.append(entry)
                done_keys.add(key)

                # 每生成一条立即落盘，防止中断丢失
                _save_output_json(out_json, results)
                done += 1
            except Exception as e:
                print(f"  ✗ 生成失败: {e}")
                failed += 1
                done += 1

    print(f"\n{'='*70}")
    print(f"✓ 批量处理完成！")
    print(f"  总推理次数: {total}")
    print(f"  成功: {done - skipped - failed}")
    print(f"  跳过（已存在）: {skipped}")
    print(f"  失败: {failed}")
    # 打印每个输出文件的记录数
    for out_path, records in output_cache.items():
        print(f"  {out_path}  ({len(records)} 条)")
    print(f"{'='*70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="视频字幕生成")
    parser.add_argument(
        "--mode", type=str, default="batch",
        choices=["batch", "single", "single_ref", "batch_ref"],
        help=(
            "运行模式: "
            "batch=批量处理文件夹, "
            "single=单个视频, "
            "single_ref=单个视频+参考图（供sh逐条调用）, "
            "batch_ref=读取任务清单一次性批量处理（模型只加载一次）"
        ),
    )
    # single_ref 模式专用参数
    parser.add_argument("--video", type=str, help="视频文件路径")
    parser.add_argument("--ref_image", type=str, help="参考角色裁剪图路径")
    parser.add_argument("--output_json", type=str, help="输出 JSON 文件路径")
    parser.add_argument("--metadata", type=str, default=None, help="角色 metadata JSON 路径")
    # batch_ref 模式专用参数
    parser.add_argument("--tasklist", type=str, help="任务清单 JSON 文件路径（batch_ref 模式）")
    parser.add_argument("--no_skip", action="store_true", help="不跳过已有结果，强制重新生成")
    # 通用参数
    parser.add_argument("--model", type=str, default=None, help="模型名称或本地路径")
    parser.add_argument("--prompt", type=str, default=None, help="自定义提示词（单条，覆盖默认）")
    # batch 模式专用参数
    parser.add_argument("--input_dir", type=str, default=VIDEO_INPUT_DIR, help="批量模式输入目录")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR, help="批量模式输出目录")

    args = parser.parse_args()
    prompts = [args.prompt] if args.prompt else PROMPT_TEXT

    if args.mode == "single_ref":
        if not args.video or not args.ref_image or not args.output_json:
            parser.error("single_ref 模式必须提供 --video, --ref_image, --output_json")
        process_single_with_ref(
            video_path=args.video,
            ref_image_path=args.ref_image,
            output_json=args.output_json,
            prompts=prompts,
            metadata_json=args.metadata,
            model_name=args.model,
        )
    elif args.mode == "batch_ref":
        if not args.tasklist:
            parser.error("batch_ref 模式必须提供 --tasklist")
        process_batch_ref(
            tasklist_json=args.tasklist,
            prompts=prompts,
            model_name=args.model,
            skip_existing=not args.no_skip,
        )
    elif args.mode == "single":
        if not args.video:
            parser.error("single 模式必须提供 --video")
        process_single_video(args.video, args.output_dir, prompts)
    else:
        process_videos_batch(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            prompt=prompts,
        )

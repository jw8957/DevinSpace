#coding=utf-8
import json
import os
import re 

from bs4 import BeautifulSoup

# To handle some legacy bug from SemanticDocument Parser
def process_markdown(originMD):
    processedMD = originMD.replace('\x00', '')
    processedMD = processedMD.replace('<list>', '').replace('</list>', '')
    nodes = processedMD.split('\\n')

    processedMD = ""
    inTableSection = False
    inCodeSection = False
    for node in nodes:
        # if '</table>' in node:
        #     print(node)
        if node.strip().startswith('#'):
            nodeStrip = node.strip()
            pt = 0
            prefix = ''
            while pt < len(nodeStrip) and nodeStrip[pt] == '#':
                prefix += '#'
                pt += 1
            node = prefix + ' ' + nodeStrip[pt:]
            # print(nodeStrip)
            # print(node)

        if node.strip().startswith('<img '):
            soup = BeautifulSoup(node, 'html.parser')
            if soup is None or soup.img is None:
                # print(node)
                processedMD += node.strip() + '\n'
            else:
                imgSrc = soup.img['src'] if 'src' in soup.img else ''
                imgAlt = soup.img['alt'] if 'alt' in soup.img else ''

                if imgSrc != '':
                    processedMD += f'![{imgAlt}]({imgSrc})\n\n'
        elif node.strip().startswith('<table>'):
            inTableSection = True
            processedMD += node.replace('<table>', '').strip() + '\n'
        elif node.strip().startswith('</table>'):
            inTableSection = False
            processedMD += node.replace('</table>', '').strip() + '\n'
            processedMD += '\n'
        
        elif node.strip().startswith('<code>'):
            inCodeSection = True
            processedMD += node.replace("<code> '''", "```").strip() + '\n'

        elif node.strip().endswith('</code>'):
            inCodeSection = False
            processedMD += node.replace("''' </code>", '```').strip() + '\n'
            processedMD += '\n'
        # elif node.strip().startswith('<list>'):
        #     print(node)
        # elif node.strip().endswith('</list>'):
        #     print(node)
        else:
            processedMD += node.strip() + '\n'
            if inTableSection == False and inCodeSection == False:
                processedMD += '\n'
    return processedMD


rm_md_image_tag_pat = re.compile(r'!\[.*?\]\((.*?)\)')
rm_html_tag_pat = re.compile(r'<[(img)|(table)].*?/?>')
def process_legacy_content(content_str, keep_img_tag = False):
    if content_str is None:
        return ''
    else:
        result = re.sub(rm_html_tag_pat, '', content_str)
        result = process_markdown(result)

        if keep_img_tag == False:
            result = re.sub(rm_md_image_tag_pat, '', result)
        
        return result


# Consolidated content filtering configuration
CONTENT_FILTERS = {
    'basic_filters': {
        'simple_matches': [
            "\n\n", "\n", "“", "”", "；", "：", "，", "。",
            "●", "|", "▲", ".", "END",
            "蓝色字", "蓝字免费订阅", "扫码关注", "免责声明",
            "当前位置", "字体 : 大 中 小",
            '【 中 】', '【 大 】', '【 小 】', '[大][中][小]',
            '【 打印文章 】'
        ],
        'phrases': [
            "\n/\n", "\n‍\n\n‍", "点击蓝字", "点蓝色字关注",
            "关注该公众号", "关注公众号", "上方名片", "下方名片",
            "点击上方", "点击下方", "图文源网络", "按住下方图片",
            "阅读原文", "下方二维码", "底部二维码", "点击二维码",
            "下面二维码", "识别二维码", "长按二维码",
            "送给朋友看看！", "分享是一种美德", "如有侵权",
            "视频源于网络", "关注后看更多", "免责申明", "版权归"
        ]
    },
    'metadata_markers': {
        'prefixes': [
            "供稿：", "供稿丨", "摄影：", "摄影丨", "初审：",
            "终审：", "终审丨", "审核：", "审核丨", "二审",
            "编辑：", "编辑丨", "编辑 丨", "来源：", "来源丨",
            "来源 |", "来源|", "来源 ", "来源/", "作者：",
            "作者丨", "声明：", "声明丨", "发布：", "发布丨"
        ],
        'common_headers': [
            ('作者：', '更新时间'),
            ('来源：',),
            ('点赞数', '分类专栏：')
        ]
    },
    'break_indicators': {
        'social': [
            ('微信', '扫描'), 
            ('微博', '微信'),
            ('关注', '公众号'), 
            ('关注', '注册'),
            ('扫描', '报名'), 
            ('扫码报名咨询',),
            ('公众号', '二维码'),
            ('Share on', 'LinkedIn'), 
            ('Share on', 'Facebook'), 
            ('Share on', 'Twitter'), 
            ('Reply With Quote')
        ],
        'navigation': [
            ('上一篇',), 
            ('下一篇',), 
            ('上一页',), 
            ('下一页',),
            ('上一个',), 
            ('下一个',),
            ('上一章',), 
            ('下一章',),
            ('Next Article',), 
            ('Previous Article',),
            ('Previous Page',), 
            ('Next Page',),
            ('Newer Post',), 
            ('Older Post',),
            ('Page', '1', '2', '3', '4'),
            ('# Related',), 
            ('# Popular',), 
            ('# Trending',),
            ('Go To Section',)
        ],
        'content_end': [
            ('本站推荐',), 
            ('最新资讯',), 
            ('最新文章',),
            ('相关阅读',), 
            ('相关问题',), 
            ('相关专题',), 
            ('相關閱讀',),
            ('推荐阅读',), 
            ('推荐视频',), 
            ('推荐新闻',),
            ('编辑推荐',), 
            ('猜你喜欢',), 
            ('更多精彩内容',),
            ('中图分类号',), 
            ('活动聚焦',),
            ('投稿指南',), 
            ('我要评论',), 
            ('谢谢浏览',),
            ('文件下载',), 
            ('附件下载',),
            ('编辑排版：',), 
            ('-  本文分类：',)
        ]
    }
}

# Unicode whitespace and invisible characters
WHITESPACE_CHARS = {
    '\u0020': 'SPACE',              # Regular space
    '\xa0': 'NO-BREAK SPACE',       # Non-breaking space (NBSP)
    '\u200b': 'ZERO WIDTH SPACE',   # Invisible but creates line break opportunities
    '\u200d': 'ZERO WIDTH JOINER'   # Used to join characters (like emojis)
}


def is_breadcrumb(contentString):
    """
    Detect if a string is a navigation breadcrumb in both Chinese and English.
    Examples:
    - 您现在的位置： 首页 > 新闻中心 > 正文
    - You are here: Home > News > Article
    - Current location: Products > Category > Detail
    - Breadcrumb: Main > Section > Page
    - Home / Categories / This Page
    """
    # Bilingual breadcrumb indicators
    BREADCRUMB_MARKERS = {
        'zh': [
            "您现在的位置",
            "当前位置",
            "位置：",
            "所在位置"
        ],
        'en': [
        ]
    }
    
    # Common navigation sections
    COMMON_SECTIONS = {
        'zh': ["首页", "正文", "详情", "内容"],
        'en': ["home", "main", "article", "detail", "page", "content"]
    }
    
    # Navigation separators (works for both languages)
    SEPARATORS = [
        " > ", ">>", " >>> ",           # Arrow style
        " / ", "/", "\\", " \\ ",       # Slash style
        " → ", "›", " › ", "»", " » ",  # Special arrows
        " - ", " | "                     # Other separators
    ]

    contentLower = contentString.lower()
    
    # Check for breadcrumb markers
    has_marker = (
        any(marker in contentString for marker in BREADCRUMB_MARKERS['zh']) or
        any(marker in contentLower for marker in BREADCRUMB_MARKERS['en'])
    )
    
    # If no marker but has common sections and separators, check pattern
    if not has_marker:
        has_common_section = (
            any(section in contentString for section in COMMON_SECTIONS['zh']) or
            any(section in contentLower for section in COMMON_SECTIONS['en'])
        )
        has_separator = any(sep in contentString for sep in SEPARATORS)
        
        # Check if it matches pattern like: "Home > Section > Page"
        if has_common_section and has_separator:
            parts = re.split('|'.join(map(re.escape, SEPARATORS)), contentString)
            # Typical breadcrumb should have 2-5 parts
            if 2 <= len(parts) <= 5:
                return True
    
    return has_marker and any(sep in contentString for sep in SEPARATORS)


def is_useful_text(contentString):
    """Check if text content is useful based on consolidated filter rules"""
    mode_length = 10
    # Normalize all special whitespace characters to regular spaces
    for char in WHITESPACE_CHARS:
        contentString = contentString.replace(char, ' ')
    
    # if len(contentString) == 0, which means is a break line, now we keep it as indicator for paragraph-breaking
    if len(contentString) == 0:
        return True

    # Check if content is a breadcrumb
    if is_breadcrumb(contentString):
        return False

    # if line only contains -, = or whitespace, skip it
    if all(c in ['-', '=', ' '] for c in contentString):
        return False

    # Check simple matches and phrases
    if contentString in CONTENT_FILTERS['basic_filters']['simple_matches']:
        return False
    
    for phrase in CONTENT_FILTERS['basic_filters']['phrases']:
        if phrase in contentString:
            return False
    
    # Check metadata prefixes
    for prefix in CONTENT_FILTERS['metadata_markers']['prefixes']:
        if prefix in contentString[0:mode_length]:
            return False
    
    return True

def RemoveMdImageTag(content):
    result = re.sub(rm_md_image_tag_pat, '', content)
    result = re.sub(r'\n{2,}', '\n\n', result)

    return result


def RephraseContent_V2(raw_markdown):

    removed_segments = []
    """Improved version of content rephrasing using consolidated filters"""
    def should_skip_line(line, idx, max_idx):
        
        # Check header markers in first few lines
        if idx < 10:
            for rule in CONTENT_FILTERS['metadata_markers']['common_headers']:
                if all(marker in line for marker in rule):
                    return True
        
        return not is_useful_text(line)

    # Define content termination categories
    CONTENT_END_CATEGORIES = {
        'social': 'Social media and promotional indicators',
        'navigation': 'Page navigation elements',
        'content_end': 'End of article markers'
    }

    def should_break_content(line, idx, max_idx):
        """Determines if content processing should stop at current line."""
        NEAR_END_THRESHOLD = 0.8  # Consider last 20% as near end
        MINIMUM_LINES_LEFT = 10    # But always keep at least 10 lines buffer
        COVERAGE_THRESHOLD = 0.5   # If matched text covers >50% of line length
        
        near_end = (idx > max_idx * NEAR_END_THRESHOLD) or (max_idx - idx < MINIMUM_LINES_LEFT)
        
        # Skip empty or very short lines
        line = line.strip()
        if not line or len(line) < 1:
            return False
            
        for category_name, _description in CONTENT_END_CATEGORIES.items():
            for pattern in CONTENT_FILTERS['break_indicators'][category_name]:
                if all(marker in line for marker in pattern):
                    # Calculate total length of matched text
                    matched_length = sum(len(marker) for marker in pattern)
                    coverage_ratio = matched_length / len(line)
                    
                    # Break if matched text covers significant portion of line
                    if coverage_ratio > COVERAGE_THRESHOLD:
                        return True
                    # Or if we're near the end of document
                    elif near_end:
                        return True
        
        return False

    # Main processing
    ### md = process_legacy_content(raw_markdown, keep_img_tag)
    md = raw_markdown
    lines = md.split('\n')
    max_line_idx = len(lines)
    clean_lines = []

    for idx, line in enumerate(lines):
        # Normalize line
        if '&nbsp' in line:
            line = line.replace('&nbsp', ' ')
            
        # Check if we should stop processing
        if should_break_content(line, idx, max_line_idx):
            removed_segments.append(f'[break from] {line}')
            break

        # Skip unwanted lines
        if should_skip_line(line, idx, max_line_idx):
            if len(line) > 0:
                removed_segments.append(f'[skip line] {line}')
            continue
            
        # Line passed all filters, keep it
        clean_lines.append(line)

    # Join lines and normalize newlines
    result = '\n'.join(clean_lines)
    result = re.sub(r'\n{2,}', '\n\n', result)
    
    result_with_img = result
    result_without_img = RemoveMdImageTag(result)

    return result_with_img, result_without_img, removed_segments

# if __name__ == '__main__':

def test_Rephrase(input_file, output_rephrase_file, output_view_file):
    # input_file = '/data/jiwei/tmp/20250206_retry/cc_sample.20250207_zh-hans_200.label.jsonl'

    # output_rephrase_file = './output_rephrase.jsonl'
    # output_view_file = './output_view.html'

    diff_list = {}
    with open(input_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            content = data['content']
            url = data['url']
            doc_uuid = data['doc_uuid']
            # lid_label = data['lid_label']

            outData = {}
            content_rephrase_with_img, content_rephrase_without_img, removed_contents  = RephraseContent_V2(content)

            diff_list[doc_uuid] = {
                'url': url,
                'origin': content,
                'rephrase_with_img': content_rephrase_with_img,
                'rephrase_without_img': content_rephrase_without_img
            }
    

    with open(output_rephrase_file, 'w') as f_out:
        for doc_uuid, diff in diff_list.items():
            f_out.write(json.dumps(diff, ensure_ascii=False) + '\n')

    # Generate a html to visualize the diff
    with open(output_view_file, 'w') as f_out:
        # Add HTML header with CSS styles
        f_out.write('''
<html>
<head>
<style>
table { 
    max-width: 95%;
    margin: 20px auto;
    border-collapse: collapse;
}
td, th {
    padding: 10px;
    vertical-align: top;
    word-wrap: break-word;
    max-width: 600px;
}
pre {
    white-space: pre-wrap;
    word-wrap: break-word;
    overflow-x: auto;
    margin: 0;
}
</style>
</head>
<body>
''')
        f_out.write('<table border="1">\n')
        f_out.write('<tr><th>Origin</th><th>Rephrase</th></tr>\n')
        for doc_uuid, diff in diff_list.items():
            f_out.write(f'<tr><td><pre>{diff["origin"]}</pre></td>')
            f_out.write(f'<td><pre>{diff["rephrase_without_img"]}</pre></td></tr>\n')

        f_out.write('</table>\n')
        f_out.write('</body></html>\n')

    
    print(f'Output diff html file: {output_view_file}')


if __name__ == '__main__':
    input_file = '/data/jiwei/tmp/20250206_retry/cc_sample.20250207_zh-hans_200.label.jsonl'
    output_rephrase_file = './output_rephrase.jsonl'
    output_view_file = './output_view.html'

    test_Rephrase(input_file, output_rephrase_file, output_view_file)
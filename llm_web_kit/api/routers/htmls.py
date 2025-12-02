"""HTML 处理路由.

提供 HTML 解析、内容提取等功能的 API 端点。
"""

import time
from typing import Optional
import base64
import html

from fastapi import (APIRouter, BackgroundTasks, Body, Depends, File,
                     HTTPException, UploadFile)
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db_manager, get_db_session
from ..dependencies import get_logger, get_settings, request_id_var
from ..models.request import HTMLParseRequest
from ..models.response import HTMLParseResponse
from ..services.html_service import HTMLService
from ..services.request_log_service import RequestLogService

logger = get_logger(__name__)
settings = get_settings()

router = APIRouter()


@router.post('/html/parse', response_model=HTMLParseResponse)
async def parse_html(
    background_tasks: BackgroundTasks,
    request: HTMLParseRequest = Body(...),
    html_service: HTMLService = Depends(HTMLService),
    db_session: Optional[AsyncSession] = Depends(get_db_session)
):
    """解析 HTML 内容.

    接收 HTML 字符串并返回解析后的结构化内容。
    """
    # 从 context var 获取 request_id
    request_id = request_id_var.get()
    decoded_bytes = base64.b64decode(request.html_content)
    decoded_str = decoded_bytes.decode('utf-8')
    unescaped_html = html.unescape(decoded_str)

    # 确定输入类型
    if request.html_content:
        input_type = 'html_content'
    elif request.url:
        input_type = 'url'
    else:
        input_type = 'unknown'

    # 创建请求日志
    start_time = time.time()
    await RequestLogService.initial_log(
        session=db_session,
        request_id=request_id,
        input_type=input_type,
        input_html=unescaped_html,
        url=request.url,
    )
    end_time = time.time()
    logger.info(f'创建日志耗时: {end_time - start_time}秒')

    try:
        logger.info(f'开始解析 HTML，内容长度: {len(unescaped_html) if unescaped_html else 0}')

        result = await html_service.parse_html(
            html_content=unescaped_html,
            url=request.url,
            request_id=request_id,
            options=request.options
        )

        # 将成功日志更新操作添加到后台任务
        background_tasks.add_task(
            RequestLogService.log_success_bg,
            request_id,
            result.get('markdown')
        )

        return HTMLParseResponse(
            success=True,
            data=result,
            message='HTML 解析成功',
            request_id=request_id
        )
    except Exception as e:
        error_message = str(e)
        logger.error(f'HTML 解析失败: {error_message}')

        # 将失败日志更新操作添加到后台任务
        background_tasks.add_task(
            RequestLogService.log_failure_bg,
            request_id,
            error_message
        )

        raise HTTPException(status_code=500, detail=f'HTML 解析失败: {error_message}')


@router.post('/html/upload')
async def upload_html_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    html_service: HTMLService = Depends(HTMLService),
    db_session: Optional[AsyncSession] = Depends(get_db_session)
):
    """上传 HTML 文件进行解析.

    支持上传 HTML 文件，自动解析并返回结果。
    """
    # 从 context var 获取 request_id
    request_id = request_id_var.get()

    try:
        if not file.filename.endswith(('.html', '.htm')):
            raise HTTPException(status_code=400, detail='只支持 HTML 文件')

        content = await file.read()
        html_content = content.decode('utf-8')

        logger.info(f'上传 HTML 文件: {file.filename}, 大小: {len(content)} bytes')
        start_time = time.time()
        # 创建请求日志
        await RequestLogService.initial_log(
            session=db_session,
            request_id=request_id,
            input_type='file',
            input_html=html_content,
            url=None,
        )
        end_time = time.time()
        logger.info(f'创建日志耗时: {end_time - start_time}秒')

        result = await html_service.parse_html(html_content=html_content, url="www.baidu.com", request_id=request_id)

        # 将成功日志更新操作添加到后台任务
        background_tasks.add_task(
            RequestLogService.log_success_bg,
            request_id,
            result.get('markdown')
        )

        return HTMLParseResponse(
            success=True,
            data=result,
            message='HTML 文件解析成功',
            request_id=request_id
        )
    except Exception as e:
        error_message = str(e)
        logger.error(f'HTML 文件解析失败: {error_message}')

        # 将失败日志更新操作添加到后台任务
        background_tasks.add_task(
            RequestLogService.log_failure_bg,
            request_id,
            error_message
        )

        raise HTTPException(status_code=500, detail=f'HTML 文件解析失败: {error_message}')


@router.get('/html/status')
async def get_service_status():
    """获取服务状态.

    返回 HTML 处理服务的当前状态信息。
    """
    return {
        'service': 'HTML Processing Service',
        'status': 'running',
        'version': '1.0.0'
    }

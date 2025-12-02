"""请求日志服务.

提供请求日志的创建、更新和查询功能。
"""

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db_manager
from ..dependencies import get_logger
from ..models.db_models import RequestLog

logger = get_logger(__name__)


class RequestLogService:
    """请求日志服务类."""
    @staticmethod
    def _generate_request_id() -> str:
        """生成唯一的请求ID."""
        return str(uuid.uuid4())

    @staticmethod
    async def create_log(
        session: Optional[AsyncSession],
        request_id: str,
        input_type: str,
        input_html: Optional[str] = None,
        url: Optional[str] = None,
    ) -> Optional[RequestLog]:
        """创建请求日志记录.

        Args:
            session: 数据库会话
            request_id: 请求ID
            input_type: 输入类型 (html_content, url, file)
            input_html: 输入HTML内容
            url: URL地址
        Returns:
            创建的日志记录，如果数据库未配置则返回 None
        """
        if session is None:
            logger.debug("数据库会话为空，跳过日志记录")
            return None
        try:
            log = RequestLog(
                request_id=request_id,
                input_type=input_type,
                input_html=input_html,
                url=url,
                status='processing',
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )
            session.add(log)
            await session.flush()  # 立即写入，获取ID
            logger.info(f"创建请求日志: request_id={request_id}, input_type={input_type}, status=processing")
            return log
        except Exception as e:
            logger.error(f"创建请求日志失败: {e}")
            return None

    @staticmethod
    async def initial_log(
        session: Optional[AsyncSession],
        request_id: str,
        input_type: str,
        input_html: Optional[str] = None,
        url: Optional[str] = None,
    ):
        """创建并提交初始日志."""
        if not session:
            logger.debug("数据库会话为空，跳过初始日志记录")
            return

        await RequestLogService.create_log(
            session, request_id, input_type, input_html, url
        )
        try:
            await session.commit()
        except Exception as e:
            logger.error(f"提交初始日志时出错: {e}")
            await session.rollback()

    @staticmethod
    async def update_log_success(
        session: Optional[AsyncSession],
        request_id: str,
        output_markdown: Optional[str] = None,
    ) -> bool:
        """更新请求日志为成功状态.

        Args:
            session: 数据库会话
            request_id: 请求ID
            output_markdown: 输出Markdown内容
        Returns:
            是否更新成功
        """
        if session is None:
            return False
        try:
            result = await session.execute(
                select(RequestLog).where(RequestLog.request_id == request_id)
            )
            log = result.scalar_one_or_none()
            if log:
                log.status = 'success'
                log.output_markdown = output_markdown
                log.updated_at = datetime.now()
                await session.flush()
                logger.info(f"更新请求日志为成功: request_id={request_id}, status=success")
                return True
            else:
                logger.warning(f"未找到请求日志: request_id={request_id}")
                return False
        except Exception as e:
            logger.error(f"更新请求日志失败: {e}")
            return False

    @staticmethod
    async def log_success_bg(request_id: str, output_markdown: Optional[str] = None):
        """作为后台任务更新日志为成功."""
        async with get_db_manager().get_session() as bg_session:
            updated = await RequestLogService.update_log_success(
                session=bg_session,
                request_id=request_id,
                output_markdown=output_markdown,
            )
            if updated:
                await bg_session.commit()

    @staticmethod
    async def update_log_failure(
        session: Optional[AsyncSession],
        request_id: str,
        error_message: str,
    ) -> bool:
        """更新请求日志为失败状态.

        Args:
            session: 数据库会话
            request_id: 请求ID
            error_message: 错误信息
        Returns:
            是否更新成功
        """
        if session is None:
            return False
        try:
            result = await session.execute(
                select(RequestLog).where(RequestLog.request_id == request_id)
            )
            log = result.scalar_one_or_none()
            if log:
                log.status = 'fail'
                log.error_message = error_message
                log.updated_at = datetime.now()
                await session.flush()
                logger.info(f"更新请求日志为失败: request_id={request_id}, status=fail")
                return True
            else:
                logger.warning(f"未找到请求日志: request_id={request_id}")
                return False

        except Exception as e:
            logger.error(f"更新请求日志失败: {e}")
            return False

    @staticmethod
    async def log_failure_bg(request_id: str, error_message: str):
        """作为后台任务更新日志为失败."""
        async with get_db_manager().get_session() as bg_session:
            updated = await RequestLogService.update_log_failure(
                session=bg_session,
                request_id=request_id,
                error_message=error_message,
            )
            if updated:
                await bg_session.commit()

    @staticmethod
    async def get_log_by_request_id(
        session: Optional[AsyncSession],
        request_id: str,
    ) -> Optional[RequestLog]:
        """根据请求ID查询日志.

        Args:
            session: 数据库会话
            request_id: 请求ID
        Returns:
            日志记录，如果未找到则返回 None
        """
        if session is None:
            return None
        try:
            result = await session.execute(
                select(RequestLog).where(RequestLog.request_id == request_id)
            )
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"查询请求日志失败: {e}")
            return None

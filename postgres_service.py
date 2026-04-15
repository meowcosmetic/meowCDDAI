import psycopg2
from psycopg2.extras import execute_values
import logging
import json
from typing import List
from models import MeowBookItem
from config import Config

logger = logging.getLogger(__name__)

class PostgresService:
    def __init__(self, 
                 host=Config.POSTGRES_HOST, 
                 port=Config.POSTGRES_PORT, 
                 dbname=Config.POSTGRES_DB, 
                 user=Config.POSTGRES_USER, 
                 password=Config.POSTGRES_PASSWORD):
        self.conn_params = {
            "host": host,
            "port": port,
            "dbname": dbname,
            "user": user,
            "password": password
        }
        self._ensure_table_exists()

    def _get_connection(self):
        return psycopg2.connect(**self.conn_params)

    def _ensure_table_exists(self):
        conn = None
        try:
            conn = self._get_connection()
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS book_contents (
                    id SERIAL PRIMARY KEY,
                    book_name TEXT,
                    chapter TEXT,
                    page FLOAT,
                    content TEXT,
                    search_queries JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            conn.commit()
            cur.close()
            logger.info("[POSTGRES] ✅ Table book_contents ensured")
        except Exception as e:
            logger.error(f"[POSTGRES] ❌ Error ensuring table: {str(e)}")
            if conn:
                conn.rollback()
        finally:
            if conn:
                conn.close()

    def insert_book_items(self, items: List[MeowBookItem]):
        if not items:
            return
        
        conn = None
        try:
            conn = self._get_connection()
            cur = conn.cursor()
            
            # Prepare data for insertion
            data = []
            for item in items:
                data.append((
                    item.Book,
                    item.Chapter,
                    item.Page,
                    item.CleanedContent or "",
                    json.dumps(item.SearchQueries or [])
                ))
            
            # Use RETURNING id to get assigned IDs
            execute_values(cur, """
                INSERT INTO book_contents (book_name, chapter, page, content, search_queries)
                VALUES %s
                RETURNING id
            """, data)
            
            ids = [row[0] for row in cur.fetchall()]
            conn.commit()
            cur.close()
            logger.info(f"[POSTGRES] ✅ Successfully inserted {len(items)} items")
            return ids
        except Exception as e:
            logger.error(f"[POSTGRES] ❌ Error inserting items: {str(e)}")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                conn.close()

    def get_neighbor_content(self, postgres_id: int):
        """
        Lấy nội dung của đoạn trước và đoạn sau dựa trên id trong cùng một cuốn sách
        """
        conn = None
        try:
            conn = self._get_connection()
            cur = conn.cursor()
            
            # Lấy thông tin sách của item hiện tại
            cur.execute("SELECT book_name FROM book_contents WHERE id = %s", (postgres_id,))
            res = cur.fetchone()
            if not res:
                return None, None
            book_name = res[0]
            
            # Tìm đoạn trước (id nhỏ hơn gần nhất trong cùng bộ sách)
            cur.execute("""
                SELECT content FROM book_contents 
                WHERE book_name = %s AND id < %s 
                ORDER BY id DESC LIMIT 1
            """, (book_name, postgres_id))
            prev_res = cur.fetchone()
            prev_content = prev_res[0] if prev_res else None
            
            # Tìm đoạn sau (id lớn hơn gần nhất trong cùng bộ sách)
            cur.execute("""
                SELECT content FROM book_contents 
                WHERE book_name = %s AND id > %s 
                ORDER BY id ASC LIMIT 1
            """, (book_name, postgres_id))
            next_res = cur.fetchone()
            next_content = next_res[0] if next_res else None
            
            cur.close()
            return prev_content, next_content
        except Exception as e:
            logger.error(f"[POSTGRES] ❌ Error getting neighbor content: {str(e)}")
            return None, None
        finally:
            if conn:
                conn.close()

# Singleton instance
postgres_service = PostgresService()

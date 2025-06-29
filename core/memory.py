import sqlite3
import json
import logging
from datetime import datetime
from typing import List, Dict, Optional, Any
import os

class MemorySystem:
    """
    SQLite-based memory system for persistent storage of creative sessions.
    Provides search, retrieval, and metadata tracking capabilities.
    """
    
    def __init__(self, db_path: str = "creative_memory.db"):
        """
        Initialize the memory system.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize the database with required tables."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create creations table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS creations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        memory_id TEXT UNIQUE NOT NULL,
                        original_prompt TEXT NOT NULL,
                        enhanced_prompt TEXT NOT NULL,
                        image_path TEXT,
                        model_3d_path TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        metadata TEXT,
                        tags TEXT
                    )
                """)
                
                # Create sessions table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS sessions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT UNIQUE NOT NULL,
                        user_id TEXT,
                        started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        creation_count INTEGER DEFAULT 0
                    )
                """)
                
                # Create indexes for better performance
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_creations_prompt ON creations(original_prompt)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_creations_created_at ON creations(created_at)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_creations_memory_id ON creations(memory_id)")
                
                conn.commit()
                logging.info("Memory database initialized successfully")
                
        except Exception as e:
            logging.error(f"Failed to initialize database: {e}")
            raise
    
    def save_creation(self, 
                     original_prompt: str, 
                     enhanced_prompt: str, 
                     image_path: Optional[str] = None,
                     model_3d_path: Optional[str] = None,
                     metadata: Optional[Dict] = None,
                     tags: Optional[List[str]] = None) -> str:
        """
        Save a creation to memory.
        
        Args:
            original_prompt: The original user prompt
            enhanced_prompt: The AI-enhanced prompt
            image_path: Path to the generated image
            model_3d_path: Path to the generated 3D model
            metadata: Additional metadata
            tags: Tags for categorization
            
        Returns:
            Memory ID for the saved creation
        """
        try:
            memory_id = self._generate_memory_id()
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO creations 
                    (memory_id, original_prompt, enhanced_prompt, image_path, model_3d_path, metadata, tags)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    memory_id,
                    original_prompt,
                    enhanced_prompt,
                    image_path,
                    model_3d_path,
                    json.dumps(metadata) if metadata else None,
                    json.dumps(tags) if tags else None
                ))
                
                conn.commit()
                logging.info(f"Creation saved with memory ID: {memory_id}")
                return memory_id
                
        except Exception as e:
            logging.error(f"Failed to save creation: {e}")
            raise
    
    def get_creation(self, memory_id: str) -> Optional[Dict]:
        """
        Retrieve a creation by memory ID.
        
        Args:
            memory_id: The memory ID to retrieve
            
        Returns:
            Creation data or None if not found
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT * FROM creations WHERE memory_id = ?
                """, (memory_id,))
                
                row = cursor.fetchone()
                if row:
                    return self._row_to_dict(row)
                return None
                
        except Exception as e:
            logging.error(f"Failed to retrieve creation: {e}")
            return None
    
    def search_creations(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Search creations by content.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of matching creations
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT * FROM creations 
                    WHERE original_prompt LIKE ? OR enhanced_prompt LIKE ? OR tags LIKE ?
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (f"%{query}%", f"%{query}%", f"%{query}%", limit))
                
                rows = cursor.fetchall()
                return [self._row_to_dict(row) for row in rows]
                
        except Exception as e:
            logging.error(f"Failed to search creations: {e}")
            return []
    
    def get_recent_creations(self, limit: int = 10) -> List[Dict]:
        """
        Get the most recent creations.
        
        Args:
            limit: Maximum number of results
            
        Returns:
            List of recent creations
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT * FROM creations 
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (limit,))
                
                rows = cursor.fetchall()
                return [self._row_to_dict(row) for row in rows]
                
        except Exception as e:
            logging.error(f"Failed to get recent creations: {e}")
            return []
    
    def get_creation_stats(self) -> Dict[str, Any]:
        """
        Get statistics about stored creations.
        
        Returns:
            Dictionary with statistics
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Total creations
                cursor.execute("SELECT COUNT(*) FROM creations")
                total_creations = cursor.fetchone()[0]
                
                # Creations with images
                cursor.execute("SELECT COUNT(*) FROM creations WHERE image_path IS NOT NULL")
                with_images = cursor.fetchone()[0]
                
                # Creations with 3D models
                cursor.execute("SELECT COUNT(*) FROM creations WHERE model_3d_path IS NOT NULL")
                with_models = cursor.fetchone()[0]
                
                # Recent activity (last 24 hours)
                cursor.execute("""
                    SELECT COUNT(*) FROM creations 
                    WHERE created_at >= datetime('now', '-1 day')
                """)
                recent_activity = cursor.fetchone()[0]
                
                return {
                    "total_creations": total_creations,
                    "with_images": with_images,
                    "with_3d_models": with_models,
                    "recent_activity_24h": recent_activity
                }
                
        except Exception as e:
            logging.error(f"Failed to get creation stats: {e}")
            return {}
    
    def get_advanced_stats(self) -> Dict[str, Any]:
        """
        Get advanced statistics and analytics.
        
        Returns:
            Dictionary with comprehensive statistics
        """
        try:
            cursor = self.conn.cursor()
            
            # Basic counts
            cursor.execute("SELECT COUNT(*) FROM creations")
            total_creations = cursor.fetchone()[0]
            
            # Time-based statistics
            cursor.execute("""
                SELECT 
                    strftime('%Y-%m-%d', timestamp) as date,
                    COUNT(*) as daily_count
                FROM creations 
                GROUP BY date 
                ORDER BY date DESC 
                LIMIT 30
            """)
            daily_stats = cursor.fetchall()
            
            # Quality analysis
            cursor.execute("""
                SELECT 
                    json_extract(metadata, '$.analysis.estimated_quality') as quality,
                    COUNT(*) as count
                FROM creations 
                WHERE json_extract(metadata, '$.analysis.estimated_quality') IS NOT NULL
                GROUP BY quality
            """)
            quality_stats = cursor.fetchall()
            
            # Complexity analysis
            cursor.execute("""
                SELECT 
                    json_extract(metadata, '$.analysis.complexity') as complexity,
                    COUNT(*) as count
                FROM creations 
                WHERE json_extract(metadata, '$.analysis.complexity') IS NOT NULL
                GROUP BY complexity
            """)
            complexity_stats = cursor.fetchall()
            
            # Processing time statistics
            cursor.execute("""
                SELECT 
                    AVG(json_extract(metadata, '$.processing_time')) as avg_time,
                    MIN(json_extract(metadata, '$.processing_time')) as min_time,
                    MAX(json_extract(metadata, '$.processing_time')) as max_time
                FROM creations 
                WHERE json_extract(metadata, '$.processing_time') IS NOT NULL
            """)
            time_stats = cursor.fetchone()
            
            # Word count statistics
            cursor.execute("""
                SELECT 
                    AVG(json_extract(metadata, '$.analysis.word_count')) as avg_words,
                    MIN(json_extract(metadata, '$.analysis.word_count')) as min_words,
                    MAX(json_extract(metadata, '$.analysis.word_count')) as max_words
                FROM creations 
                WHERE json_extract(metadata, '$.analysis.word_count') IS NOT NULL
            """)
            word_stats = cursor.fetchone()
            
            # Most common words in prompts
            cursor.execute("""
                SELECT 
                    LOWER(word) as word,
                    COUNT(*) as frequency
                FROM creations 
                CROSS JOIN json_each('["' || REPLACE(original_prompt, ' ', '","') || '"]') as words(word)
                WHERE LENGTH(word) > 2
                GROUP BY LOWER(word)
                ORDER BY frequency DESC
                LIMIT 20
            """)
            word_frequency = cursor.fetchall()
            
            # Success rate
            cursor.execute("""
                SELECT 
                    CASE 
                        WHEN image_path LIKE '%fallback%' THEN 'fallback'
                        ELSE 'real'
                    END as generation_type,
                    COUNT(*) as count
                FROM creations 
                GROUP BY generation_type
            """)
            generation_stats = cursor.fetchall()
            
            return {
                "total_creations": total_creations,
                "daily_stats": [{"date": date, "count": count} for date, count in daily_stats],
                "quality_distribution": [{"quality": quality, "count": count} for quality, count in quality_stats],
                "complexity_distribution": [{"complexity": complexity, "count": count} for complexity, count in complexity_stats],
                "processing_time": {
                    "average": time_stats[0] if time_stats[0] else 0,
                    "minimum": time_stats[1] if time_stats[1] else 0,
                    "maximum": time_stats[2] if time_stats[2] else 0
                },
                "word_count": {
                    "average": word_stats[0] if word_stats[0] else 0,
                    "minimum": word_stats[1] if word_stats[1] else 0,
                    "maximum": word_stats[2] if word_stats[2] else 0
                },
                "most_common_words": [{"word": word, "frequency": freq} for word, freq in word_frequency],
                "generation_types": [{"type": gen_type, "count": count} for gen_type, count in generation_stats]
            }
            
        except Exception as e:
            logger.error(f"Failed to get advanced stats: {e}")
            return {"error": str(e)}

    def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """
        Get statistics for a specific user.
        
        Args:
            user_id: The user ID to get stats for
            
        Returns:
            Dictionary with user-specific statistics
        """
        try:
            cursor = self.conn.cursor()
            
            # User-specific counts
            cursor.execute("""
                SELECT COUNT(*) FROM creations 
                WHERE json_extract(metadata, '$.user_id') = ?
            """, (user_id,))
            user_creations = cursor.fetchone()[0]
            
            # User's recent activity
            cursor.execute("""
                SELECT 
                    strftime('%Y-%m-%d', timestamp) as date,
                    COUNT(*) as daily_count
                FROM creations 
                WHERE json_extract(metadata, '$.user_id') = ?
                GROUP BY date 
                ORDER BY date DESC 
                LIMIT 7
            """, (user_id,))
            user_daily_stats = cursor.fetchall()
            
            # User's average processing time
            cursor.execute("""
                SELECT AVG(json_extract(metadata, '$.processing_time'))
                FROM creations 
                WHERE json_extract(metadata, '$.user_id') = ?
                AND json_extract(metadata, '$.processing_time') IS NOT NULL
            """, (user_id,))
            user_avg_time = cursor.fetchone()[0]
            
            return {
                "user_id": user_id,
                "total_creations": user_creations,
                "daily_activity": [{"date": date, "count": count} for date, count in user_daily_stats],
                "average_processing_time": user_avg_time if user_avg_time else 0
            }
            
        except Exception as e:
            logger.error(f"Failed to get user stats: {e}")
            return {"error": str(e)}

    def get_trends(self, days: int = 30) -> Dict[str, Any]:
        """
        Get trends over a specified number of days.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Dictionary with trend data
        """
        try:
            cursor = self.conn.cursor()
            
            # Daily creation trends
            cursor.execute("""
                SELECT 
                    strftime('%Y-%m-%d', timestamp) as date,
                    COUNT(*) as creations,
                    AVG(json_extract(metadata, '$.processing_time')) as avg_time,
                    AVG(json_extract(metadata, '$.analysis.word_count')) as avg_words
                FROM creations 
                WHERE timestamp >= datetime('now', '-{} days')
                GROUP BY date 
                ORDER BY date
            """.format(days))
            daily_trends = cursor.fetchall()
            
            # Quality trends
            cursor.execute("""
                SELECT 
                    strftime('%Y-%m-%d', timestamp) as date,
                    json_extract(metadata, '$.analysis.estimated_quality') as quality,
                    COUNT(*) as count
                FROM creations 
                WHERE timestamp >= datetime('now', '-{} days')
                AND json_extract(metadata, '$.analysis.estimated_quality') IS NOT NULL
                GROUP BY date, quality
                ORDER BY date
            """.format(days))
            quality_trends = cursor.fetchall()
            
            return {
                "period_days": days,
                "daily_trends": [
                    {
                        "date": date,
                        "creations": creations,
                        "avg_processing_time": avg_time if avg_time else 0,
                        "avg_word_count": avg_words if avg_words else 0
                    }
                    for date, creations, avg_time, avg_words in daily_trends
                ],
                "quality_trends": [
                    {
                        "date": date,
                        "quality": quality,
                        "count": count
                    }
                    for date, quality, count in quality_trends
                ]
            }
            
        except Exception as e:
            logger.error(f"Failed to get trends: {e}")
            return {"error": str(e)}
    
    def _generate_memory_id(self) -> str:
        """Generate a unique memory ID."""
        import uuid
        return str(uuid.uuid4()).replace('-', '')[:12]
    
    def _row_to_dict(self, row: tuple) -> Dict:
        """Convert database row to dictionary."""
        columns = [
            'id', 'memory_id', 'original_prompt', 'enhanced_prompt',
            'image_path', 'model_3d_path', 'created_at', 'metadata', 'tags'
        ]
        
        result = dict(zip(columns, row))
        
        # Parse JSON fields
        if result['metadata']:
            try:
                result['metadata'] = json.loads(result['metadata'])
            except:
                result['metadata'] = {}
        
        if result['tags']:
            try:
                result['tags'] = json.loads(result['tags'])
            except:
                result['tags'] = []
        
        return result
    
    def cleanup_orphaned_files(self) -> int:
        """
        Clean up orphaned files that are referenced in the database but don't exist.
        
        Returns:
            Number of orphaned files found
        """
        orphaned_count = 0
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get all file paths
                cursor.execute("SELECT image_path, model_3d_path FROM creations")
                rows = cursor.fetchall()
                
                for row in rows:
                    image_path, model_3d_path = row
                    
                    # Check image file
                    if image_path and not os.path.exists(image_path):
                        logging.warning(f"Orphaned image file: {image_path}")
                        orphaned_count += 1
                    
                    # Check 3D model file
                    if model_3d_path and not os.path.exists(model_3d_path):
                        logging.warning(f"Orphaned 3D model file: {model_3d_path}")
                        orphaned_count += 1
                
                return orphaned_count
                
        except Exception as e:
            logging.error(f"Failed to cleanup orphaned files: {e}")
            return 0 
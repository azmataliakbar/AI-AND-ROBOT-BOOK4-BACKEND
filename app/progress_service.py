from typing import Optional
from .models import ProgressTracker as ProgressTrackerModel, User, Chapter
from .database import SessionLocal
from .schemas import ProgressTrackerCreate, ProgressTrackerResponse
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ProgressTrackingService:
    def __init__(self):
        pass

    def get_progress(self, user_id: str, chapter_id: str) -> Optional[ProgressTrackerResponse]:
        """
        Get the progress for a specific user and chapter
        """
        db: Session = SessionLocal()
        try:
            progress = db.query(ProgressTrackerModel).filter(
                ProgressTrackerModel.user_id == user_id,
                ProgressTrackerModel.chapter_id == chapter_id
            ).first()

            if progress:
                return ProgressTrackerResponse(
                    id=progress.id,
                    user_id=progress.user_id,
                    chapter_id=progress.chapter_id,
                    completion_percentage=progress.completion_percentage,
                    last_read_position=progress.last_read_position,
                    time_spent=progress.time_spent,
                    completed_at=progress.completed_at,
                    created_at=progress.created_at,
                    updated_at=progress.updated_at
                )
            return None
        finally:
            db.close()

    def update_progress(self, progress_data: ProgressTrackerCreate) -> Optional[ProgressTrackerResponse]:
        """
        Update or create progress for a user and chapter
        """
        db: Session = SessionLocal()
        try:
            # Check if progress record already exists
            existing_progress = db.query(ProgressTrackerModel).filter(
                ProgressTrackerModel.user_id == progress_data.user_id,
                ProgressTrackerModel.chapter_id == progress_data.chapter_id
            ).first()

            if existing_progress:
                # Update existing record
                existing_progress.completion_percentage = progress_data.completion_percentage
                existing_progress.last_read_position = progress_data.last_read_position
                existing_progress.time_spent = progress_data.time_spent
                existing_progress.updated_at = datetime.utcnow()

                # Mark as completed if percentage reaches 100
                if progress_data.completion_percentage >= 100 and not existing_progress.completed_at:
                    existing_progress.completed_at = datetime.utcnow()
            else:
                # Create new record
                progress_record = ProgressTrackerModel(
                    id=f"progress_{datetime.utcnow().timestamp()}_{progress_data.user_id[:8]}_{progress_data.chapter_id[:8]}",
                    user_id=progress_data.user_id,
                    chapter_id=progress_data.chapter_id,
                    completion_percentage=progress_data.completion_percentage,
                    last_read_position=progress_data.last_read_position,
                    time_spent=progress_data.time_spent,
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow()
                )

                # Mark as completed if percentage is 100
                if progress_data.completion_percentage >= 100:
                    progress_record.completed_at = datetime.utcnow()

                db.add(progress_record)

            db.commit()

            # Return the updated/created record
            progress = db.query(ProgressTrackerModel).filter(
                ProgressTrackerModel.user_id == progress_data.user_id,
                ProgressTrackerModel.chapter_id == progress_data.chapter_id
            ).first()

            return ProgressTrackerResponse(
                id=progress.id,
                user_id=progress.user_id,
                chapter_id=progress.chapter_id,
                completion_percentage=progress.completion_percentage,
                last_read_position=progress.last_read_position,
                time_spent=progress.time_spent,
                completed_at=progress.completed_at,
                created_at=progress.created_at,
                updated_at=progress.updated_at
            )
        except IntegrityError:
            db.rollback()
            logger.error(f"Integrity error while updating progress for user {progress_data.user_id} and chapter {progress_data.chapter_id}")
            return None
        except Exception as e:
            db.rollback()
            logger.error(f"Error updating progress: {str(e)}")
            return None
        finally:
            db.close()

    def get_user_progress_summary(self, user_id: str) -> dict:
        """
        Get a summary of user's progress across all chapters
        """
        db: Session = SessionLocal()
        try:
            # Get all progress records for the user
            progress_records = db.query(ProgressTrackerModel).filter(
                ProgressTrackerModel.user_id == user_id
            ).all()

            if not progress_records:
                return {
                    "user_id": user_id,
                    "total_chapters": 0,
                    "completed_chapters": 0,
                    "in_progress_chapters": 0,
                    "overall_progress": 0.0,
                    "total_time_spent": 0
                }

            total_chapters = len(progress_records)
            completed_chapters = sum(1 for p in progress_records if p.completion_percentage >= 100)
            in_progress_chapters = sum(1 for p in progress_records if 0 < p.completion_percentage < 100)
            overall_progress = sum(p.completion_percentage for p in progress_records) / total_chapters if total_chapters > 0 else 0
            total_time_spent = sum(p.time_spent for p in progress_records)

            return {
                "user_id": user_id,
                "total_chapters": total_chapters,
                "completed_chapters": completed_chapters,
                "in_progress_chapters": in_progress_chapters,
                "overall_progress": round(overall_progress, 2),
                "total_time_spent": total_time_spent
            }
        finally:
            db.close()


# Global instance
progress_service = ProgressTrackingService()
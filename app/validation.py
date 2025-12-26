import re
from typing import Optional
from pydantic import BaseModel, validator, Field
from bleach import clean


class InputValidator:
    """Utility class for validating and sanitizing user inputs"""

    @staticmethod
    def sanitize_text(text: str) -> str:
        """Sanitize text input by removing potentially harmful HTML tags and attributes"""
        # Use bleach to sanitize HTML content
        sanitized = clean(
            text,
            tags=[],  # Allow no HTML tags
            attributes={},  # Allow no HTML attributes
            strip=True  # Remove disallowed tags
        )
        return sanitized.strip()

    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email format"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None

    @staticmethod
    def validate_username(username: str) -> bool:
        """Validate username format (alphanumeric and underscores only, 3-30 chars)"""
        if len(username) < 3 or len(username) > 30:
            return False
        pattern = r'^[a-zA-Z0-9_]+$'
        return re.match(pattern, username) is not None

    @staticmethod
    def validate_query(query: str) -> tuple[bool, str]:
        """Validate query length and content"""
        if not query or len(query.strip()) == 0:
            return False, "Query cannot be empty"

        if len(query) > 2000:
            return False, "Query too long. Maximum 2000 characters allowed"

        # Check for potentially harmful content
        harmful_patterns = [
            r'<script',  # Script tags
            r'javascript:',  # JavaScript URLs
            r'on\w+\s*=',  # Event handlers
            r'eval\s*\(',  # Eval function
            r'expression\s*\('  # Expression function
        ]

        for pattern in harmful_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return False, "Query contains potentially harmful content"

        return True, ""

    @staticmethod
    def validate_chapter_id(chapter_id: str) -> tuple[bool, str]:
        """Validate chapter ID format"""
        if not chapter_id:
            return False, "Chapter ID cannot be empty"

        # Expected format: ch_XXX where XXX is 3 digits
        pattern = r'^ch_\d{3}$'
        if not re.match(pattern, chapter_id):
            return False, "Invalid chapter ID format. Expected format: ch_XXX"

        return True, ""

    @staticmethod
    def validate_user_id(user_id: str) -> tuple[bool, str]:
        """Validate user ID format"""
        if not user_id:
            return False, "User ID cannot be empty"

        # Expected format: user_XXXXXXXX where X is alphanumeric
        pattern = r'^user_[a-zA-Z0-9]+$'
        if not re.match(pattern, user_id):
            return False, "Invalid user ID format"

        return True, ""

    @staticmethod
    def validate_completion_percentage(percentage: float) -> tuple[bool, str]:
        """Validate completion percentage"""
        if percentage < 0.0 or percentage > 100.0:
            return False, "Completion percentage must be between 0.0 and 100.0"

        return True, ""


# Enhanced schemas with validation
class ValidatedChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    user_id: Optional[str] = None
    chapter_id: Optional[str] = None

    def sanitizied_query(self) -> str:
        """Return sanitized query"""
        return InputValidator.sanitize_text(self.query)

    def sanitizied_user_id(self) -> Optional[str]:
        """Return sanitized user_id if valid"""
        if self.user_id:
            is_valid, _ = InputValidator.validate_user_id(self.user_id)
            if is_valid:
                return InputValidator.sanitize_text(self.user_id)
        return None

    def sanitizied_chapter_id(self) -> Optional[str]:
        """Return sanitized chapter_id if valid"""
        if self.chapter_id:
            is_valid, _ = InputValidator.validate_chapter_id(self.chapter_id)
            if is_valid:
                return InputValidator.sanitize_text(self.chapter_id)
        return None

    @validator('query')
    def validate_query_field(cls, v):
        is_valid, error_msg = InputValidator.validate_query(v)
        if not is_valid:
            raise ValueError(error_msg)
        return v

    @validator('user_id')
    def validate_user_id_field(cls, v):
        if v is not None:
            is_valid, error_msg = InputValidator.validate_user_id(v)
            if not is_valid:
                raise ValueError(error_msg)
        return v

    @validator('chapter_id')
    def validate_chapter_id_field(cls, v):
        if v is not None:
            is_valid, error_msg = InputValidator.validate_chapter_id(v)
            if not is_valid:
                raise ValueError(error_msg)
        return v


class ValidatedProgressTrackerCreate(BaseModel):
    user_id: str
    chapter_id: str
    completion_percentage: float = Field(ge=0.0, le=100.0)
    last_read_position: int = Field(ge=0)
    time_spent: int = Field(ge=0)

    @validator('user_id')
    def validate_user_id_field(cls, v):
        is_valid, error_msg = InputValidator.validate_user_id(v)
        if not is_valid:
            raise ValueError(error_msg)
        return v

    @validator('chapter_id')
    def validate_chapter_id_field(cls, v):
        is_valid, error_msg = InputValidator.validate_chapter_id(v)
        if not is_valid:
            raise ValueError(error_msg)
        return v
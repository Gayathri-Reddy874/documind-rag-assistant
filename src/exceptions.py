"""Domain-specific exceptions for clear, actionable error handling."""


class DocumentAssistantError(Exception):
    """Base exception for all application-level errors."""


class UnsupportedFileTypeError(DocumentAssistantError):
    """Raised when a user uploads a file type that isn't supported."""


class FileTooLargeError(DocumentAssistantError):
    """Raised when an uploaded file exceeds the configured size limit."""


class EmptyDocumentError(DocumentAssistantError):
    """Raised when a document contains no extractable text."""


class VectorStoreNotReadyError(DocumentAssistantError):
    """Raised when a query is issued before any document has been indexed."""


class LLMConfigurationError(DocumentAssistantError):
    """Raised when the LLM provider is misconfigured (e.g. missing API key)."""

"""OAuth Proxy Service for simplified user authentication.

This service handles complex OAuth flows so users only need to click "Connect" buttons.
Features:
- Automatic OAuth 2.0 flow handling
- Token refresh automation
- Secure token storage
- Platform-specific OAuth configurations
"""

import secrets
from datetime import datetime, timedelta
from typing import Any, Dict, Optional
from urllib.parse import urlencode
from uuid import UUID

import aiohttp
from pydantic import BaseModel

from app.core.credential_vault import CredentialVault, CredentialType
from app.core.errors import ConnectorError
from app.core.models import SourcePlatform


class OAuthConfig(BaseModel):
    """OAuth configuration for a platform."""
    
    platform: SourcePlatform
    client_id: str
    client_secret: str
    authorization_url: str
    token_url: str
    scopes: list[str]
    redirect_uri: str


class OAuthState(BaseModel):
    """OAuth state for CSRF protection."""
    
    state: str
    user_id: UUID
    platform: SourcePlatform
    created_at: datetime
    expires_at: datetime


class OAuthProxyService:
    """OAuth proxy service for simplified authentication.
    
    This service abstracts away OAuth complexity from users.
    Users just click "Connect [Platform]" and we handle the rest.
    """

    # Platform-specific OAuth configurations
    OAUTH_CONFIGS = {
        SourcePlatform.REDDIT: {
            "authorization_url": "https://www.reddit.com/api/v1/authorize",
            "token_url": "https://www.reddit.com/api/v1/access_token",
            "scopes": ["identity", "read", "mysubreddits", "history"],
        },
        SourcePlatform.YOUTUBE: {
            "authorization_url": "https://accounts.google.com/o/oauth2/v2/auth",
            "token_url": "https://oauth2.googleapis.com/token",
            "scopes": [
                "https://www.googleapis.com/auth/youtube.readonly",
                "https://www.googleapis.com/auth/youtube.force-ssl"
            ],
        },
        SourcePlatform.TIKTOK: {
            "authorization_url": "https://www.tiktok.com/v2/auth/authorize/",
            "token_url": "https://open.tiktokapis.com/v2/oauth/token/",
            "scopes": ["user.info.basic", "video.list"],
        },
        SourcePlatform.FACEBOOK: {
            "authorization_url": "https://www.facebook.com/v21.0/dialog/oauth",
            "token_url": "https://graph.facebook.com/v21.0/oauth/access_token",
            "scopes": ["public_profile", "user_posts", "pages_read_engagement"],
        },
        SourcePlatform.INSTAGRAM: {
            "authorization_url": "https://api.instagram.com/oauth/authorize",
            "token_url": "https://api.instagram.com/oauth/access_token",
            "scopes": ["user_profile", "user_media"],
        },
        SourcePlatform.WECHAT: {
            "authorization_url": "https://open.weixin.qq.com/connect/oauth2/authorize",
            "token_url": "https://api.weixin.qq.com/sns/oauth2/access_token",
            "scopes": ["snsapi_userinfo"],
        },
    }

    def __init__(
        self,
        credential_vault: CredentialVault,
        app_credentials: Dict[SourcePlatform, Dict[str, str]],
        base_redirect_uri: str
    ):
        """Initialize OAuth proxy service.
        
        Args:
            credential_vault: Credential vault for secure storage
            app_credentials: Platform app credentials (client_id, client_secret)
            base_redirect_uri: Base redirect URI (e.g., "https://app.example.com/oauth/callback")
        """
        self.vault = credential_vault
        self.app_credentials = app_credentials
        self.base_redirect_uri = base_redirect_uri
        
        # In-memory state storage (in production, use Redis)
        self.oauth_states: Dict[str, OAuthState] = {}

    def get_authorization_url(
        self,
        user_id: UUID,
        platform: SourcePlatform
    ) -> str:
        """Generate authorization URL for user to connect platform.
        
        Args:
            user_id: User ID
            platform: Platform to connect
            
        Returns:
            Authorization URL for user to visit
        """
        if platform not in self.OAUTH_CONFIGS:
            raise ConnectorError(f"OAuth not supported for {platform}")
        
        if platform not in self.app_credentials:
            raise ConnectorError(f"App credentials not configured for {platform}")
        
        config = self.OAUTH_CONFIGS[platform]
        app_creds = self.app_credentials[platform]
        
        # Generate state for CSRF protection
        state = secrets.token_urlsafe(32)
        self.oauth_states[state] = OAuthState(
            state=state,
            user_id=user_id,
            platform=platform,
            created_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(minutes=10)
        )
        
        # Build authorization URL
        params = {
            "client_id": app_creds["client_id"],
            "redirect_uri": f"{self.base_redirect_uri}/{platform.value}",
            "scope": " ".join(config["scopes"]),
            "state": state,
            "response_type": "code",
        }
        
        # Platform-specific parameters
        if platform == SourcePlatform.REDDIT:
            params["duration"] = "permanent"
        elif platform == SourcePlatform.YOUTUBE:
            params["access_type"] = "offline"
            params["prompt"] = "consent"
        
        return f"{config['authorization_url']}?{urlencode(params)}"

    async def handle_callback(
        self,
        platform: SourcePlatform,
        code: str,
        state: str,
        user_password: str
    ) -> Dict[str, Any]:
        """Handle OAuth callback and exchange code for tokens.
        
        Args:
            platform: Platform
            code: Authorization code
            state: State parameter (for CSRF protection)
            user_password: User's password for credential encryption
            
        Returns:
            Token information
            
        Raises:
            ConnectorError: If callback handling fails
        """
        # Verify state
        if state not in self.oauth_states:
            raise ConnectorError("Invalid or expired state")
        
        oauth_state = self.oauth_states[state]
        
        # Check expiration
        if datetime.utcnow() > oauth_state.expires_at:
            del self.oauth_states[state]
            raise ConnectorError("State expired")
        
        # Verify platform matches
        if oauth_state.platform != platform:
            raise ConnectorError("Platform mismatch")
        
        # Exchange code for tokens
        tokens = await self._exchange_code_for_tokens(platform, code)
        
        # Store tokens securely
        credential_id = await self.vault.store_credential(
            user_id=oauth_state.user_id,
            platform=platform.value,
            credential_type=CredentialType.OAUTH_TOKEN,
            credential_data=tokens,
            user_password=user_password,
            requires_mfa=False
        )
        
        # Clean up state
        del self.oauth_states[state]
        
        return {
            "credential_id": str(credential_id),
            "platform": platform.value,
            "expires_at": tokens.get("expires_at"),
        }

    async def _exchange_code_for_tokens(
        self,
        platform: SourcePlatform,
        code: str
    ) -> Dict[str, Any]:
        """Exchange authorization code for access tokens.
        
        Args:
            platform: Platform
            code: Authorization code
            
        Returns:
            Token data (access_token, refresh_token, expires_in, etc.)
        """
        config = self.OAUTH_CONFIGS[platform]
        app_creds = self.app_credentials[platform]
        
        data = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": f"{self.base_redirect_uri}/{platform.value}",
            "client_id": app_creds["client_id"],
            "client_secret": app_creds["client_secret"],
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(config["token_url"], data=data) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise ConnectorError(f"Token exchange failed: {error_text}")
                
                tokens = await response.json()
        
        # Calculate expiration time
        if "expires_in" in tokens:
            tokens["expires_at"] = (
                datetime.utcnow() + timedelta(seconds=tokens["expires_in"])
            ).isoformat()
        
        return tokens

    async def refresh_token(
        self,
        credential_id: UUID,
        user_password: str
    ) -> Dict[str, Any]:
        """Refresh OAuth token.
        
        Args:
            credential_id: Credential ID
            user_password: User's password
            
        Returns:
            New token data
        """
        # Retrieve current tokens
        tokens = await self.vault.retrieve_credential(credential_id, user_password)
        
        if "refresh_token" not in tokens:
            raise ConnectorError("No refresh token available")
        
        # Get credential metadata to determine platform
        from sqlalchemy import select
        from app.core.credential_vault import EncryptedCredential
        
        result = await self.vault.db.execute(
            select(EncryptedCredential).where(EncryptedCredential.id == credential_id)
        )
        credential = result.scalar_one_or_none()
        
        if not credential:
            raise ConnectorError("Credential not found")
        
        platform = SourcePlatform(credential.platform)
        config = self.OAUTH_CONFIGS[platform]
        app_creds = self.app_credentials[platform]
        
        # Refresh token
        data = {
            "grant_type": "refresh_token",
            "refresh_token": tokens["refresh_token"],
            "client_id": app_creds["client_id"],
            "client_secret": app_creds["client_secret"],
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(config["token_url"], data=data) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise ConnectorError(f"Token refresh failed: {error_text}")
                
                new_tokens = await response.json()
        
        # Preserve refresh token if not returned
        if "refresh_token" not in new_tokens:
            new_tokens["refresh_token"] = tokens["refresh_token"]
        
        # Calculate expiration
        if "expires_in" in new_tokens:
            new_tokens["expires_at"] = (
                datetime.utcnow() + timedelta(seconds=new_tokens["expires_in"])
            ).isoformat()
        
        # Update stored credential
        await self.vault.rotate_credential(
            credential_id,
            user_password,
            new_tokens
        )
        
        return new_tokens


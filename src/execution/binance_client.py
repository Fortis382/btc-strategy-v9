# src/execution/binance_client.py
"""
Binance API 래퍼 (v9.4 실행 레이어)
"""
from __future__ import annotations
from typing import Dict, Any, Optional, List
import requests
import time
import hmac
import hashlib
from urllib.parse import urlencode

class BinanceClient:
    """
    Binance Futures API 클라이언트
    
    기능:
        - 주문 생성/취소
        - OHLCV 조회
        - 계좌 정보
    
    참고:
        - API 문서: https://binance-docs.github.io/apidocs/futures/en/
        - Rate Limit: 1200 req/min (IP), 50 orders/10s
    """
    
    def __init__(
        self,
        api_key: str = "",
        secret_key: str = "",
        testnet: bool = True
    ):
        self.api_key = api_key
        self.secret_key = secret_key
        
        if testnet:
            self.base_url = "https://testnet.binancefuture.com"
        else:
            self.base_url = "https://fapi.binance.com"
    
    def _sign(self, params: Dict[str, Any]) -> str:
        """HMAC SHA256 서명"""
        query_string = urlencode(params)
        signature = hmac.new(
            self.secret_key.encode(),
            query_string.encode(),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        signed: bool = False
    ) -> Dict[str, Any]:
        """HTTP 요청"""
        if params is None:
            params = {}
        
        headers = {"X-MBX-APIKEY": self.api_key}
        
        if signed:
            params["timestamp"] = int(time.time() * 1000)
            params["signature"] = self._sign(params)
        
        url = f"{self.base_url}{endpoint}"
        
        if method == "GET":
            response = requests.get(url, params=params, headers=headers)
        elif method == "POST":
            response = requests.post(url, params=params, headers=headers)
        elif method == "DELETE":
            response = requests.delete(url, params=params, headers=headers)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        response.raise_for_status()
        return response.json()
    
    def get_klines(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "15m",
        limit: int = 500
    ) -> List[List]:
        """
        OHLCV 조회
        
        Returns:
            [[open_time, open, high, low, close, volume, ...], ...]
        """
        endpoint = "/fapi/v1/klines"
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
        }
        
        return self._request("GET", endpoint, params)
    
    def place_futures_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: str = "GTC"
    ) -> Dict[str, Any]:
        """
        선물 주문
        
        Args:
            symbol: "BTCUSDT"
            side: "BUY" | "SELL"
            order_type: "MARKET" | "LIMIT" | "STOP" | "STOP_MARKET"
            quantity: 수량
            price: 지정가 (LIMIT)
            stop_price: 스톱가 (STOP)
            time_in_force: "GTC" (Good Till Cancel)
        
        Returns:
            주문 응답
        """
        endpoint = "/fapi/v1/order"
        params = {
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "quantity": quantity,
            "timeInForce": time_in_force,
        }
        
        if price:
            params["price"] = price
        if stop_price:
            params["stopPrice"] = stop_price
        
        return self._request("POST", endpoint, params, signed=True)
    
    def cancel_order(
        self,
        symbol: str,
        order_id: str
    ) -> Dict[str, Any]:
        """주문 취소"""
        endpoint = "/fapi/v1/order"
        params = {
            "symbol": symbol,
            "orderId": order_id,
        }
        
        return self._request("DELETE", endpoint, params, signed=True)
    
    def get_account(self) -> Dict[str, Any]:
        """계좌 정보"""
        endpoint = "/fapi/v2/account"
        return self._request("GET", endpoint, signed=True)
    
    def get_position(self, symbol: str = "BTCUSDT") -> Dict[str, Any]:
        """포지션 조회"""
        endpoint = "/fapi/v2/positionRisk"
        params = {"symbol": symbol}
        return self._request("GET", endpoint, params, signed=True)
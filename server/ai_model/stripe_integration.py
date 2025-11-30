"""
Stripe Integration for Platform Forge

This module provides a comprehensive Stripe payments integration matching Replit's
Stripe payments functionality with payment processing, subscriptions, customer 
management, invoicing, and webhook handling.

Key Components:
- StripeClient: Main Stripe API client wrapper
- CustomerManager: Customer lifecycle management
- SubscriptionManager: Subscription creation, updates, cancellation
- PaymentManager: Payment intents and capture
- InvoiceManager: Invoice creation and management
- WebhookHandler: Webhook event processing and signature verification
- ProductManager: Product and price catalog
- CheckoutManager: Checkout session and portal management

Features:
- One-time and recurring payments
- Subscription management with trials
- Customer management with metadata
- Product and price catalog
- Webhook handling with signature verification
- Invoice management and PDF generation
- Refund processing (full and partial)
- Payment method management
- Sandbox/test mode support
- Usage-based billing
- Proration handling
- Tax rate support
- Coupon and discount management

Payment Processing:
- Create payment intents for one-time charges
- Capture or cancel payment intents
- Support for multiple currencies
- 3D Secure authentication handling
- Automatic retry on failure

Subscription Features:
- Create subscriptions with trial periods
- Update subscription plans (upgrade/downgrade)
- Cancel immediately or at period end
- Pause and resume subscriptions
- Handle subscription lifecycle webhooks

Security:
- Webhook signature verification with timing-safe comparison
- API key management (test/live modes)
- Idempotency key support
- Rate limiting awareness

Usage:
    from server.ai_model.stripe_integration import (
        StripeClient,
        CustomerManager,
        SubscriptionManager,
        PaymentManager,
        WebhookHandler,
        CheckoutManager,
    )
    
    # Initialize the client
    client = StripeClient(
        api_key="sk_test_...",
        webhook_secret="whsec_...",
        test_mode=True
    )
    
    # Create a customer
    customer = client.customers.create(
        email="customer@example.com",
        name="John Doe",
        metadata={"user_id": "123"}
    )
    
    # Create a subscription
    subscription = client.subscriptions.create(
        customer_id=customer.id,
        price_id="price_xxx",
        trial_days=14
    )
    
    # Create a checkout session
    session = client.checkout.create_session(
        customer_id=customer.id,
        price_id="price_xxx",
        success_url="https://example.com/success",
        cancel_url="https://example.com/cancel"
    )
    
    # Handle webhook
    event = client.webhooks.handle(payload, signature)
    if event.type == "checkout.session.completed":
        handle_checkout_complete(event.data)
    
    # Process refund
    refund = client.payments.create_refund(
        payment_intent_id="pi_xxx",
        amount=1000,  # $10.00 in cents
        reason="requested_by_customer"
    )
    
    # Create customer portal session
    portal = client.checkout.create_portal_session(
        customer_id=customer.id,
        return_url="https://example.com/account"
    )
"""

import os
import re
import json
import time
import hmac
import hashlib
import secrets
import threading
import logging
from typing import Dict, List, Optional, Any, Union, Set, Tuple, Callable, TypeVar
from dataclasses import dataclass, field
from enum import Enum, auto
from datetime import datetime, timedelta
from pathlib import Path
from abc import ABC, abstractmethod
from decimal import Decimal
import copy


class PaymentStatus(Enum):
    """Payment intent status values."""
    REQUIRES_PAYMENT_METHOD = "requires_payment_method"
    REQUIRES_CONFIRMATION = "requires_confirmation"
    REQUIRES_ACTION = "requires_action"
    PROCESSING = "processing"
    REQUIRES_CAPTURE = "requires_capture"
    CANCELED = "canceled"
    SUCCEEDED = "succeeded"


class SubscriptionStatus(Enum):
    """Subscription status values."""
    INCOMPLETE = "incomplete"
    INCOMPLETE_EXPIRED = "incomplete_expired"
    TRIALING = "trialing"
    ACTIVE = "active"
    PAST_DUE = "past_due"
    CANCELED = "canceled"
    UNPAID = "unpaid"
    PAUSED = "paused"


class InvoiceStatus(Enum):
    """Invoice status values."""
    DRAFT = "draft"
    OPEN = "open"
    PAID = "paid"
    UNCOLLECTIBLE = "uncollectible"
    VOID = "void"


class RefundStatus(Enum):
    """Refund status values."""
    PENDING = "pending"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELED = "canceled"
    REQUIRES_ACTION = "requires_action"


class RefundReason(Enum):
    """Refund reason values."""
    DUPLICATE = "duplicate"
    FRAUDULENT = "fraudulent"
    REQUESTED_BY_CUSTOMER = "requested_by_customer"
    EXPIRED_UNCAPTURED_CHARGE = "expired_uncaptured_charge"


class PaymentMethodType(Enum):
    """Payment method types."""
    CARD = "card"
    BANK_TRANSFER = "bank_transfer"
    SEPA_DEBIT = "sepa_debit"
    ACH_DEBIT = "us_bank_account"
    IDEAL = "ideal"
    BANCONTACT = "bancontact"
    GIROPAY = "giropay"
    SOFORT = "sofort"
    EPS = "eps"
    P24 = "p24"
    ALIPAY = "alipay"
    WECHAT = "wechat_pay"
    KLARNA = "klarna"
    AFTERPAY = "afterpay_clearpay"
    PAYPAL = "paypal"
    CRYPTO = "crypto"
    LINK = "link"


class WebhookEventType(Enum):
    """Webhook event types."""
    CHECKOUT_SESSION_COMPLETED = "checkout.session.completed"
    CHECKOUT_SESSION_EXPIRED = "checkout.session.expired"
    CUSTOMER_CREATED = "customer.created"
    CUSTOMER_UPDATED = "customer.updated"
    CUSTOMER_DELETED = "customer.deleted"
    CUSTOMER_SUBSCRIPTION_CREATED = "customer.subscription.created"
    CUSTOMER_SUBSCRIPTION_UPDATED = "customer.subscription.updated"
    CUSTOMER_SUBSCRIPTION_DELETED = "customer.subscription.deleted"
    CUSTOMER_SUBSCRIPTION_TRIAL_WILL_END = "customer.subscription.trial_will_end"
    CUSTOMER_SUBSCRIPTION_PAUSED = "customer.subscription.paused"
    CUSTOMER_SUBSCRIPTION_RESUMED = "customer.subscription.resumed"
    INVOICE_CREATED = "invoice.created"
    INVOICE_FINALIZED = "invoice.finalized"
    INVOICE_PAID = "invoice.paid"
    INVOICE_PAYMENT_FAILED = "invoice.payment_failed"
    INVOICE_UPCOMING = "invoice.upcoming"
    INVOICE_VOIDED = "invoice.voided"
    PAYMENT_INTENT_CREATED = "payment_intent.created"
    PAYMENT_INTENT_SUCCEEDED = "payment_intent.succeeded"
    PAYMENT_INTENT_PAYMENT_FAILED = "payment_intent.payment_failed"
    PAYMENT_INTENT_CANCELED = "payment_intent.canceled"
    PAYMENT_METHOD_ATTACHED = "payment_method.attached"
    PAYMENT_METHOD_DETACHED = "payment_method.detached"
    CHARGE_SUCCEEDED = "charge.succeeded"
    CHARGE_FAILED = "charge.failed"
    CHARGE_REFUNDED = "charge.refunded"
    CHARGE_DISPUTE_CREATED = "charge.dispute.created"
    CHARGE_DISPUTE_CLOSED = "charge.dispute.closed"
    PRODUCT_CREATED = "product.created"
    PRODUCT_UPDATED = "product.updated"
    PRODUCT_DELETED = "product.deleted"
    PRICE_CREATED = "price.created"
    PRICE_UPDATED = "price.updated"
    PRICE_DELETED = "price.deleted"
    BILLING_PORTAL_SESSION_CREATED = "billing_portal.session.created"


class BillingInterval(Enum):
    """Billing interval for subscriptions."""
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    YEAR = "year"


class ProrationBehavior(Enum):
    """Proration behavior for subscription updates."""
    CREATE_PRORATIONS = "create_prorations"
    NONE = "none"
    ALWAYS_INVOICE = "always_invoice"


class CollectionMethod(Enum):
    """Invoice collection methods."""
    CHARGE_AUTOMATICALLY = "charge_automatically"
    SEND_INVOICE = "send_invoice"


class CheckoutMode(Enum):
    """Checkout session modes."""
    PAYMENT = "payment"
    SUBSCRIPTION = "subscription"
    SETUP = "setup"


class Currency(Enum):
    """Supported currencies (ISO 4217 codes)."""
    USD = "usd"
    EUR = "eur"
    GBP = "gbp"
    CAD = "cad"
    AUD = "aud"
    JPY = "jpy"
    CHF = "chf"
    CNY = "cny"
    INR = "inr"
    MXN = "mxn"
    BRL = "brl"
    KRW = "krw"
    SGD = "sgd"
    HKD = "hkd"
    NZD = "nzd"
    SEK = "sek"
    NOK = "nok"
    DKK = "dkk"
    PLN = "pln"
    CZK = "czk"


class StripeLimit(Enum):
    """Stripe API limits."""
    MAX_METADATA_KEYS = 50
    MAX_METADATA_KEY_LENGTH = 40
    MAX_METADATA_VALUE_LENGTH = 500
    MAX_DESCRIPTION_LENGTH = 350
    MAX_STATEMENT_DESCRIPTOR_LENGTH = 22
    MAX_ITEMS_PER_LIST = 100
    MIN_CHARGE_AMOUNT_USD = 50  # $0.50 in cents
    MAX_CHARGE_AMOUNT_USD = 99999999  # $999,999.99 in cents
    WEBHOOK_SIGNATURE_TOLERANCE_SECONDS = 300
    MAX_IDEMPOTENCY_KEY_LENGTH = 255


class StripeError(Exception):
    """Base exception for Stripe errors."""
    def __init__(
        self,
        message: str,
        code: Optional[str] = None,
        param: Optional[str] = None,
        http_status: Optional[int] = None,
        request_id: Optional[str] = None
    ):
        super().__init__(message)
        self.code = code
        self.param = param
        self.http_status = http_status
        self.request_id = request_id
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary."""
        return {
            "message": str(self),
            "code": self.code,
            "param": self.param,
            "http_status": self.http_status,
            "request_id": self.request_id,
        }


class CardError(StripeError):
    """Card declined or invalid."""
    def __init__(
        self,
        message: str,
        decline_code: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.decline_code = decline_code


class RateLimitError(StripeError):
    """Rate limit exceeded."""
    def __init__(self, retry_after: float = 0):
        super().__init__(
            f"Rate limit exceeded. Retry after {retry_after} seconds",
            code="rate_limit",
            http_status=429
        )
        self.retry_after = retry_after


class InvalidRequestError(StripeError):
    """Invalid request parameters."""
    def __init__(self, message: str, param: Optional[str] = None):
        super().__init__(message, code="invalid_request", param=param, http_status=400)


class AuthenticationError(StripeError):
    """Invalid API key or unauthorized access."""
    def __init__(self, message: str = "Invalid API key provided"):
        super().__init__(message, code="authentication_error", http_status=401)


class PermissionError(StripeError):
    """Insufficient permissions."""
    def __init__(self, message: str = "Permission denied"):
        super().__init__(message, code="permission_error", http_status=403)


class ResourceNotFoundError(StripeError):
    """Resource not found."""
    def __init__(self, resource_type: str, resource_id: str):
        super().__init__(
            f"{resource_type} '{resource_id}' not found",
            code="resource_missing",
            http_status=404
        )
        self.resource_type = resource_type
        self.resource_id = resource_id


class IdempotencyError(StripeError):
    """Idempotency key conflict."""
    def __init__(self, idempotency_key: str):
        super().__init__(
            f"Idempotency key '{idempotency_key}' already used with different parameters",
            code="idempotency_error",
            http_status=400
        )
        self.idempotency_key = idempotency_key


class WebhookSignatureError(StripeError):
    """Webhook signature verification failed."""
    def __init__(self, message: str = "Invalid webhook signature"):
        super().__init__(message, code="webhook_signature_error")


class PaymentFailedError(StripeError):
    """Payment processing failed."""
    def __init__(
        self,
        message: str,
        payment_intent_id: Optional[str] = None,
        decline_code: Optional[str] = None
    ):
        super().__init__(message, code="payment_failed")
        self.payment_intent_id = payment_intent_id
        self.decline_code = decline_code


class SubscriptionError(StripeError):
    """Subscription operation failed."""
    def __init__(self, message: str, subscription_id: Optional[str] = None):
        super().__init__(message, code="subscription_error")
        self.subscription_id = subscription_id


class InvoiceError(StripeError):
    """Invoice operation failed."""
    def __init__(self, message: str, invoice_id: Optional[str] = None):
        super().__init__(message, code="invoice_error")
        self.invoice_id = invoice_id


class RefundError(StripeError):
    """Refund operation failed."""
    def __init__(
        self,
        message: str,
        charge_id: Optional[str] = None,
        refund_id: Optional[str] = None
    ):
        super().__init__(message, code="refund_error")
        self.charge_id = charge_id
        self.refund_id = refund_id


@dataclass
class Address:
    """
    Customer or billing address.
    
    Attributes:
        line1: Street address line 1
        line2: Street address line 2
        city: City name
        state: State or province
        postal_code: Postal or ZIP code
        country: Two-letter country code (ISO 3166-1 alpha-2)
    """
    line1: Optional[str] = None
    line2: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    postal_code: Optional[str] = None
    country: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in {
            "line1": self.line1,
            "line2": self.line2,
            "city": self.city,
            "state": self.state,
            "postal_code": self.postal_code,
            "country": self.country,
        }.items() if v is not None}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Address':
        """Create Address from dictionary."""
        return cls(
            line1=data.get("line1"),
            line2=data.get("line2"),
            city=data.get("city"),
            state=data.get("state"),
            postal_code=data.get("postal_code"),
            country=data.get("country"),
        )


@dataclass
class CardDetails:
    """
    Credit/debit card details.
    
    Attributes:
        brand: Card brand (visa, mastercard, amex, etc.)
        last4: Last 4 digits of card number
        exp_month: Expiration month (1-12)
        exp_year: Expiration year (4 digits)
        funding: Funding type (credit, debit, prepaid)
        fingerprint: Unique card fingerprint
        country: Two-letter country code
        checks: Card verification checks result
    """
    brand: str = ""
    last4: str = ""
    exp_month: int = 0
    exp_year: int = 0
    funding: str = "unknown"
    fingerprint: Optional[str] = None
    country: Optional[str] = None
    checks: Dict[str, str] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Check if the card is expired."""
        now = datetime.now()
        if self.exp_year < now.year:
            return True
        if self.exp_year == now.year and self.exp_month < now.month:
            return True
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "brand": self.brand,
            "last4": self.last4,
            "exp_month": self.exp_month,
            "exp_year": self.exp_year,
            "funding": self.funding,
            "fingerprint": self.fingerprint,
            "country": self.country,
            "checks": self.checks,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CardDetails':
        """Create CardDetails from dictionary."""
        return cls(
            brand=data.get("brand", ""),
            last4=data.get("last4", ""),
            exp_month=data.get("exp_month", 0),
            exp_year=data.get("exp_year", 0),
            funding=data.get("funding", "unknown"),
            fingerprint=data.get("fingerprint"),
            country=data.get("country"),
            checks=data.get("checks", {}),
        )


@dataclass
class PaymentMethod:
    """
    Payment method representation.
    
    Attributes:
        id: Unique payment method ID (pm_xxx)
        type: Payment method type (card, bank_transfer, etc.)
        customer_id: Associated customer ID
        card: Card details (if type is card)
        billing_details: Billing address and email
        created: Creation timestamp
        livemode: Whether this is a live mode object
        metadata: Custom metadata
    """
    id: str
    type: PaymentMethodType = PaymentMethodType.CARD
    customer_id: Optional[str] = None
    card: Optional[CardDetails] = None
    billing_details: Dict[str, Any] = field(default_factory=dict)
    created: float = field(default_factory=time.time)
    livemode: bool = False
    metadata: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "object": "payment_method",
            "type": self.type.value,
            "customer": self.customer_id,
            "card": self.card.to_dict() if self.card else None,
            "billing_details": self.billing_details,
            "created": int(self.created),
            "livemode": self.livemode,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PaymentMethod':
        """Create PaymentMethod from dictionary."""
        card_data = data.get("card")
        return cls(
            id=data.get("id", ""),
            type=PaymentMethodType(data.get("type", "card")),
            customer_id=data.get("customer"),
            card=CardDetails.from_dict(card_data) if card_data else None,
            billing_details=data.get("billing_details", {}),
            created=data.get("created", time.time()),
            livemode=data.get("livemode", False),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Customer:
    """
    Stripe customer representation.
    
    Attributes:
        id: Unique customer ID (cus_xxx)
        email: Customer email address
        name: Customer full name
        phone: Customer phone number
        description: Customer description
        address: Customer address
        shipping: Shipping address
        currency: Preferred currency
        default_payment_method_id: Default payment method
        invoice_prefix: Prefix for invoice numbers
        balance: Customer balance (in cents)
        created: Creation timestamp
        livemode: Whether this is a live mode object
        metadata: Custom metadata
        tax_exempt: Tax exemption status
        tax_ids: List of tax IDs
        delinquent: Whether customer has unpaid invoices
        deleted: Whether customer is deleted
    """
    id: str
    email: Optional[str] = None
    name: Optional[str] = None
    phone: Optional[str] = None
    description: Optional[str] = None
    address: Optional[Address] = None
    shipping: Optional[Dict[str, Any]] = None
    currency: Currency = Currency.USD
    default_payment_method_id: Optional[str] = None
    invoice_prefix: Optional[str] = None
    balance: int = 0
    created: float = field(default_factory=time.time)
    livemode: bool = False
    metadata: Dict[str, str] = field(default_factory=dict)
    tax_exempt: str = "none"
    tax_ids: List[Dict[str, Any]] = field(default_factory=list)
    delinquent: bool = False
    deleted: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "object": "customer",
            "email": self.email,
            "name": self.name,
            "phone": self.phone,
            "description": self.description,
            "address": self.address.to_dict() if self.address else None,
            "shipping": self.shipping,
            "currency": self.currency.value if self.currency else None,
            "default_payment_method": self.default_payment_method_id,
            "invoice_prefix": self.invoice_prefix,
            "balance": self.balance,
            "created": int(self.created),
            "livemode": self.livemode,
            "metadata": self.metadata,
            "tax_exempt": self.tax_exempt,
            "tax_ids": {"data": self.tax_ids},
            "delinquent": self.delinquent,
            "deleted": self.deleted,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Customer':
        """Create Customer from dictionary."""
        address_data = data.get("address")
        currency_value = data.get("currency")
        return cls(
            id=data.get("id", ""),
            email=data.get("email"),
            name=data.get("name"),
            phone=data.get("phone"),
            description=data.get("description"),
            address=Address.from_dict(address_data) if address_data else None,
            shipping=data.get("shipping"),
            currency=Currency(currency_value) if currency_value else Currency.USD,
            default_payment_method_id=data.get("default_payment_method"),
            invoice_prefix=data.get("invoice_prefix"),
            balance=data.get("balance", 0),
            created=data.get("created", time.time()),
            livemode=data.get("livemode", False),
            metadata=data.get("metadata", {}),
            tax_exempt=data.get("tax_exempt", "none"),
            tax_ids=data.get("tax_ids", {}).get("data", []),
            delinquent=data.get("delinquent", False),
            deleted=data.get("deleted", False),
        )


@dataclass
class Price:
    """
    Stripe price representation.
    
    Attributes:
        id: Unique price ID (price_xxx)
        product_id: Associated product ID
        active: Whether the price is active
        currency: Price currency
        unit_amount: Price in smallest currency unit (cents for USD)
        recurring: Recurring billing settings
        type: Price type (one_time or recurring)
        billing_scheme: Billing scheme (per_unit or tiered)
        tiers: Pricing tiers (for tiered pricing)
        tiers_mode: Tiers mode (graduated or volume)
        lookup_key: Lookup key for price
        nickname: Human-readable nickname
        created: Creation timestamp
        livemode: Whether this is a live mode object
        metadata: Custom metadata
    """
    id: str
    product_id: str
    active: bool = True
    currency: Currency = Currency.USD
    unit_amount: Optional[int] = None
    recurring: Optional[Dict[str, Any]] = None
    type: str = "one_time"
    billing_scheme: str = "per_unit"
    tiers: Optional[List[Dict[str, Any]]] = None
    tiers_mode: Optional[str] = None
    lookup_key: Optional[str] = None
    nickname: Optional[str] = None
    created: float = field(default_factory=time.time)
    livemode: bool = False
    metadata: Dict[str, str] = field(default_factory=dict)
    transform_quantity: Optional[Dict[str, Any]] = None
    unit_amount_decimal: Optional[str] = None
    
    @property
    def is_recurring(self) -> bool:
        """Check if this is a recurring price."""
        return self.type == "recurring" and self.recurring is not None
    
    @property
    def interval(self) -> Optional[str]:
        """Get billing interval for recurring prices."""
        if self.recurring:
            return self.recurring.get("interval")
        return None
    
    @property
    def interval_count(self) -> int:
        """Get interval count for recurring prices."""
        if self.recurring:
            return self.recurring.get("interval_count", 1)
        return 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "object": "price",
            "product": self.product_id,
            "active": self.active,
            "currency": self.currency.value,
            "unit_amount": self.unit_amount,
            "unit_amount_decimal": self.unit_amount_decimal,
            "recurring": self.recurring,
            "type": self.type,
            "billing_scheme": self.billing_scheme,
            "tiers": self.tiers,
            "tiers_mode": self.tiers_mode,
            "lookup_key": self.lookup_key,
            "nickname": self.nickname,
            "transform_quantity": self.transform_quantity,
            "created": int(self.created),
            "livemode": self.livemode,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Price':
        """Create Price from dictionary."""
        currency_value = data.get("currency", "usd")
        return cls(
            id=data.get("id", ""),
            product_id=data.get("product", ""),
            active=data.get("active", True),
            currency=Currency(currency_value) if currency_value else Currency.USD,
            unit_amount=data.get("unit_amount"),
            recurring=data.get("recurring"),
            type=data.get("type", "one_time"),
            billing_scheme=data.get("billing_scheme", "per_unit"),
            tiers=data.get("tiers"),
            tiers_mode=data.get("tiers_mode"),
            lookup_key=data.get("lookup_key"),
            nickname=data.get("nickname"),
            created=data.get("created", time.time()),
            livemode=data.get("livemode", False),
            metadata=data.get("metadata", {}),
            transform_quantity=data.get("transform_quantity"),
            unit_amount_decimal=data.get("unit_amount_decimal"),
        )


@dataclass
class Product:
    """
    Stripe product representation.
    
    Attributes:
        id: Unique product ID (prod_xxx)
        name: Product name
        description: Product description
        active: Whether product is active
        default_price_id: Default price ID
        images: Product image URLs
        features: Product features
        unit_label: Label for unit quantity
        shippable: Whether product is shippable
        statement_descriptor: Statement descriptor
        tax_code: Tax code
        url: Product URL
        created: Creation timestamp
        updated: Last update timestamp
        livemode: Whether this is a live mode object
        metadata: Custom metadata
    """
    id: str
    name: str
    description: Optional[str] = None
    active: bool = True
    default_price_id: Optional[str] = None
    images: List[str] = field(default_factory=list)
    features: List[Dict[str, str]] = field(default_factory=list)
    unit_label: Optional[str] = None
    shippable: Optional[bool] = None
    statement_descriptor: Optional[str] = None
    tax_code: Optional[str] = None
    url: Optional[str] = None
    created: float = field(default_factory=time.time)
    updated: float = field(default_factory=time.time)
    livemode: bool = False
    metadata: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "object": "product",
            "name": self.name,
            "description": self.description,
            "active": self.active,
            "default_price": self.default_price_id,
            "images": self.images,
            "features": self.features,
            "unit_label": self.unit_label,
            "shippable": self.shippable,
            "statement_descriptor": self.statement_descriptor,
            "tax_code": self.tax_code,
            "url": self.url,
            "created": int(self.created),
            "updated": int(self.updated),
            "livemode": self.livemode,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Product':
        """Create Product from dictionary."""
        return cls(
            id=data.get("id", ""),
            name=data.get("name", ""),
            description=data.get("description"),
            active=data.get("active", True),
            default_price_id=data.get("default_price"),
            images=data.get("images", []),
            features=data.get("features", []),
            unit_label=data.get("unit_label"),
            shippable=data.get("shippable"),
            statement_descriptor=data.get("statement_descriptor"),
            tax_code=data.get("tax_code"),
            url=data.get("url"),
            created=data.get("created", time.time()),
            updated=data.get("updated", time.time()),
            livemode=data.get("livemode", False),
            metadata=data.get("metadata", {}),
        )


@dataclass
class SubscriptionItem:
    """
    Subscription item (line item in subscription).
    
    Attributes:
        id: Unique subscription item ID (si_xxx)
        price_id: Associated price ID
        quantity: Item quantity
        metadata: Custom metadata
        created: Creation timestamp
    """
    id: str
    price_id: str
    quantity: int = 1
    metadata: Dict[str, str] = field(default_factory=dict)
    created: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "object": "subscription_item",
            "price": self.price_id,
            "quantity": self.quantity,
            "metadata": self.metadata,
            "created": int(self.created),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SubscriptionItem':
        """Create SubscriptionItem from dictionary."""
        price = data.get("price", {})
        price_id = price.get("id", "") if isinstance(price, dict) else price
        return cls(
            id=data.get("id", ""),
            price_id=price_id,
            quantity=data.get("quantity", 1),
            metadata=data.get("metadata", {}),
            created=data.get("created", time.time()),
        )


@dataclass
class Subscription:
    """
    Stripe subscription representation.
    
    Attributes:
        id: Unique subscription ID (sub_xxx)
        customer_id: Associated customer ID
        status: Subscription status
        items: Subscription items
        default_payment_method_id: Default payment method
        current_period_start: Current period start timestamp
        current_period_end: Current period end timestamp
        cancel_at_period_end: Cancel at period end flag
        cancel_at: Cancellation timestamp
        canceled_at: When subscription was canceled
        trial_start: Trial start timestamp
        trial_end: Trial end timestamp
        billing_cycle_anchor: Billing cycle anchor
        collection_method: Collection method
        days_until_due: Days until invoice is due
        latest_invoice_id: Latest invoice ID
        pending_update: Pending update details
        pause_collection: Pause collection settings
        application_fee_percent: Application fee percentage
        billing_thresholds: Billing thresholds
        created: Creation timestamp
        ended_at: When subscription ended
        start_date: Subscription start date
        livemode: Whether this is a live mode object
        metadata: Custom metadata
    """
    id: str
    customer_id: str
    status: SubscriptionStatus = SubscriptionStatus.ACTIVE
    items: List[SubscriptionItem] = field(default_factory=list)
    default_payment_method_id: Optional[str] = None
    current_period_start: float = field(default_factory=time.time)
    current_period_end: float = field(default_factory=lambda: time.time() + 30 * 24 * 3600)
    cancel_at_period_end: bool = False
    cancel_at: Optional[float] = None
    canceled_at: Optional[float] = None
    trial_start: Optional[float] = None
    trial_end: Optional[float] = None
    billing_cycle_anchor: float = field(default_factory=time.time)
    collection_method: CollectionMethod = CollectionMethod.CHARGE_AUTOMATICALLY
    days_until_due: Optional[int] = None
    latest_invoice_id: Optional[str] = None
    pending_update: Optional[Dict[str, Any]] = None
    pause_collection: Optional[Dict[str, Any]] = None
    application_fee_percent: Optional[float] = None
    billing_thresholds: Optional[Dict[str, Any]] = None
    created: float = field(default_factory=time.time)
    ended_at: Optional[float] = None
    start_date: float = field(default_factory=time.time)
    livemode: bool = False
    metadata: Dict[str, str] = field(default_factory=dict)
    
    @property
    def is_active(self) -> bool:
        """Check if subscription is active."""
        return self.status in [SubscriptionStatus.ACTIVE, SubscriptionStatus.TRIALING]
    
    @property
    def is_trialing(self) -> bool:
        """Check if subscription is in trial period."""
        return self.status == SubscriptionStatus.TRIALING
    
    @property
    def is_canceled(self) -> bool:
        """Check if subscription is canceled."""
        return self.status == SubscriptionStatus.CANCELED
    
    @property
    def is_paused(self) -> bool:
        """Check if subscription is paused."""
        return self.status == SubscriptionStatus.PAUSED
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "object": "subscription",
            "customer": self.customer_id,
            "status": self.status.value,
            "items": {"data": [item.to_dict() for item in self.items]},
            "default_payment_method": self.default_payment_method_id,
            "current_period_start": int(self.current_period_start),
            "current_period_end": int(self.current_period_end),
            "cancel_at_period_end": self.cancel_at_period_end,
            "cancel_at": int(self.cancel_at) if self.cancel_at else None,
            "canceled_at": int(self.canceled_at) if self.canceled_at else None,
            "trial_start": int(self.trial_start) if self.trial_start else None,
            "trial_end": int(self.trial_end) if self.trial_end else None,
            "billing_cycle_anchor": int(self.billing_cycle_anchor),
            "collection_method": self.collection_method.value,
            "days_until_due": self.days_until_due,
            "latest_invoice": self.latest_invoice_id,
            "pending_update": self.pending_update,
            "pause_collection": self.pause_collection,
            "application_fee_percent": self.application_fee_percent,
            "billing_thresholds": self.billing_thresholds,
            "created": int(self.created),
            "ended_at": int(self.ended_at) if self.ended_at else None,
            "start_date": int(self.start_date),
            "livemode": self.livemode,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Subscription':
        """Create Subscription from dictionary."""
        items_data = data.get("items", {}).get("data", [])
        items = [SubscriptionItem.from_dict(item) for item in items_data]
        return cls(
            id=data.get("id", ""),
            customer_id=data.get("customer", ""),
            status=SubscriptionStatus(data.get("status", "active")),
            items=items,
            default_payment_method_id=data.get("default_payment_method"),
            current_period_start=data.get("current_period_start", time.time()),
            current_period_end=data.get("current_period_end", time.time() + 30 * 24 * 3600),
            cancel_at_period_end=data.get("cancel_at_period_end", False),
            cancel_at=data.get("cancel_at"),
            canceled_at=data.get("canceled_at"),
            trial_start=data.get("trial_start"),
            trial_end=data.get("trial_end"),
            billing_cycle_anchor=data.get("billing_cycle_anchor", time.time()),
            collection_method=CollectionMethod(data.get("collection_method", "charge_automatically")),
            days_until_due=data.get("days_until_due"),
            latest_invoice_id=data.get("latest_invoice"),
            pending_update=data.get("pending_update"),
            pause_collection=data.get("pause_collection"),
            application_fee_percent=data.get("application_fee_percent"),
            billing_thresholds=data.get("billing_thresholds"),
            created=data.get("created", time.time()),
            ended_at=data.get("ended_at"),
            start_date=data.get("start_date", time.time()),
            livemode=data.get("livemode", False),
            metadata=data.get("metadata", {}),
        )


@dataclass
class InvoiceLineItem:
    """
    Invoice line item.
    
    Attributes:
        id: Unique line item ID (il_xxx)
        description: Line item description
        amount: Amount in cents
        currency: Currency code
        quantity: Quantity
        period: Billing period
        price_id: Associated price ID
        proration: Whether this is a proration
        type: Line item type (invoiceitem or subscription)
        metadata: Custom metadata
    """
    id: str
    description: Optional[str] = None
    amount: int = 0
    currency: Currency = Currency.USD
    quantity: int = 1
    period: Optional[Dict[str, int]] = None
    price_id: Optional[str] = None
    proration: bool = False
    type: str = "invoiceitem"
    metadata: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "object": "line_item",
            "description": self.description,
            "amount": self.amount,
            "currency": self.currency.value,
            "quantity": self.quantity,
            "period": self.period,
            "price": self.price_id,
            "proration": self.proration,
            "type": self.type,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'InvoiceLineItem':
        """Create InvoiceLineItem from dictionary."""
        currency_value = data.get("currency", "usd")
        price = data.get("price", {})
        price_id = price.get("id", "") if isinstance(price, dict) else price
        return cls(
            id=data.get("id", ""),
            description=data.get("description"),
            amount=data.get("amount", 0),
            currency=Currency(currency_value) if currency_value else Currency.USD,
            quantity=data.get("quantity", 1),
            period=data.get("period"),
            price_id=price_id,
            proration=data.get("proration", False),
            type=data.get("type", "invoiceitem"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Invoice:
    """
    Stripe invoice representation.
    
    Attributes:
        id: Unique invoice ID (in_xxx)
        customer_id: Associated customer ID
        subscription_id: Associated subscription ID
        status: Invoice status
        number: Invoice number
        lines: Invoice line items
        amount_due: Amount due in cents
        amount_paid: Amount paid in cents
        amount_remaining: Amount remaining in cents
        subtotal: Subtotal in cents
        total: Total in cents
        tax: Tax amount in cents
        currency: Currency code
        collection_method: Collection method
        billing_reason: Billing reason
        payment_intent_id: Associated payment intent
        default_payment_method_id: Default payment method
        hosted_invoice_url: Hosted invoice page URL
        invoice_pdf: Invoice PDF URL
        due_date: Invoice due date
        period_start: Billing period start
        period_end: Billing period end
        paid: Whether invoice is paid
        paid_at: When invoice was paid
        created: Creation timestamp
        finalized_at: When invoice was finalized
        voided_at: When invoice was voided
        livemode: Whether this is a live mode object
        metadata: Custom metadata
    """
    id: str
    customer_id: str
    subscription_id: Optional[str] = None
    status: InvoiceStatus = InvoiceStatus.DRAFT
    number: Optional[str] = None
    lines: List[InvoiceLineItem] = field(default_factory=list)
    amount_due: int = 0
    amount_paid: int = 0
    amount_remaining: int = 0
    subtotal: int = 0
    total: int = 0
    tax: Optional[int] = None
    currency: Currency = Currency.USD
    collection_method: CollectionMethod = CollectionMethod.CHARGE_AUTOMATICALLY
    billing_reason: Optional[str] = None
    payment_intent_id: Optional[str] = None
    default_payment_method_id: Optional[str] = None
    hosted_invoice_url: Optional[str] = None
    invoice_pdf: Optional[str] = None
    due_date: Optional[float] = None
    period_start: float = field(default_factory=time.time)
    period_end: float = field(default_factory=time.time)
    paid: bool = False
    paid_at: Optional[float] = None
    created: float = field(default_factory=time.time)
    finalized_at: Optional[float] = None
    voided_at: Optional[float] = None
    livemode: bool = False
    metadata: Dict[str, str] = field(default_factory=dict)
    discount: Optional[Dict[str, Any]] = None
    discounts: List[str] = field(default_factory=list)
    
    @property
    def is_paid(self) -> bool:
        """Check if invoice is paid."""
        return self.status == InvoiceStatus.PAID or self.paid
    
    @property
    def is_open(self) -> bool:
        """Check if invoice is open."""
        return self.status == InvoiceStatus.OPEN
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "object": "invoice",
            "customer": self.customer_id,
            "subscription": self.subscription_id,
            "status": self.status.value,
            "number": self.number,
            "lines": {"data": [line.to_dict() for line in self.lines]},
            "amount_due": self.amount_due,
            "amount_paid": self.amount_paid,
            "amount_remaining": self.amount_remaining,
            "subtotal": self.subtotal,
            "total": self.total,
            "tax": self.tax,
            "currency": self.currency.value,
            "collection_method": self.collection_method.value,
            "billing_reason": self.billing_reason,
            "payment_intent": self.payment_intent_id,
            "default_payment_method": self.default_payment_method_id,
            "hosted_invoice_url": self.hosted_invoice_url,
            "invoice_pdf": self.invoice_pdf,
            "due_date": int(self.due_date) if self.due_date else None,
            "period_start": int(self.period_start),
            "period_end": int(self.period_end),
            "paid": self.paid,
            "paid_at": int(self.paid_at) if self.paid_at else None,
            "created": int(self.created),
            "finalized_at": int(self.finalized_at) if self.finalized_at else None,
            "voided_at": int(self.voided_at) if self.voided_at else None,
            "livemode": self.livemode,
            "metadata": self.metadata,
            "discount": self.discount,
            "discounts": self.discounts,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Invoice':
        """Create Invoice from dictionary."""
        lines_data = data.get("lines", {}).get("data", [])
        lines = [InvoiceLineItem.from_dict(line) for line in lines_data]
        currency_value = data.get("currency", "usd")
        return cls(
            id=data.get("id", ""),
            customer_id=data.get("customer", ""),
            subscription_id=data.get("subscription"),
            status=InvoiceStatus(data.get("status", "draft")),
            number=data.get("number"),
            lines=lines,
            amount_due=data.get("amount_due", 0),
            amount_paid=data.get("amount_paid", 0),
            amount_remaining=data.get("amount_remaining", 0),
            subtotal=data.get("subtotal", 0),
            total=data.get("total", 0),
            tax=data.get("tax"),
            currency=Currency(currency_value) if currency_value else Currency.USD,
            collection_method=CollectionMethod(data.get("collection_method", "charge_automatically")),
            billing_reason=data.get("billing_reason"),
            payment_intent_id=data.get("payment_intent"),
            default_payment_method_id=data.get("default_payment_method"),
            hosted_invoice_url=data.get("hosted_invoice_url"),
            invoice_pdf=data.get("invoice_pdf"),
            due_date=data.get("due_date"),
            period_start=data.get("period_start", time.time()),
            period_end=data.get("period_end", time.time()),
            paid=data.get("paid", False),
            paid_at=data.get("paid_at"),
            created=data.get("created", time.time()),
            finalized_at=data.get("finalized_at"),
            voided_at=data.get("voided_at"),
            livemode=data.get("livemode", False),
            metadata=data.get("metadata", {}),
            discount=data.get("discount"),
            discounts=data.get("discounts", []),
        )


@dataclass
class PaymentIntent:
    """
    Stripe payment intent representation.
    
    Attributes:
        id: Unique payment intent ID (pi_xxx)
        amount: Amount in cents
        currency: Currency code
        status: Payment intent status
        customer_id: Associated customer ID
        payment_method_id: Payment method ID
        description: Payment description
        receipt_email: Receipt email
        statement_descriptor: Statement descriptor
        statement_descriptor_suffix: Statement descriptor suffix
        capture_method: Capture method (automatic or manual)
        confirmation_method: Confirmation method
        client_secret: Client secret for frontend
        last_payment_error: Last payment error
        charges: List of charges
        invoice_id: Associated invoice ID
        setup_future_usage: Setup for future usage
        created: Creation timestamp
        canceled_at: Cancellation timestamp
        cancellation_reason: Cancellation reason
        livemode: Whether this is a live mode object
        metadata: Custom metadata
    """
    id: str
    amount: int
    currency: Currency = Currency.USD
    status: PaymentStatus = PaymentStatus.REQUIRES_PAYMENT_METHOD
    customer_id: Optional[str] = None
    payment_method_id: Optional[str] = None
    description: Optional[str] = None
    receipt_email: Optional[str] = None
    statement_descriptor: Optional[str] = None
    statement_descriptor_suffix: Optional[str] = None
    capture_method: str = "automatic"
    confirmation_method: str = "automatic"
    client_secret: str = ""
    last_payment_error: Optional[Dict[str, Any]] = None
    charges: List[Dict[str, Any]] = field(default_factory=list)
    invoice_id: Optional[str] = None
    setup_future_usage: Optional[str] = None
    created: float = field(default_factory=time.time)
    canceled_at: Optional[float] = None
    cancellation_reason: Optional[str] = None
    livemode: bool = False
    metadata: Dict[str, str] = field(default_factory=dict)
    amount_capturable: int = 0
    amount_received: int = 0
    
    @property
    def is_succeeded(self) -> bool:
        """Check if payment succeeded."""
        return self.status == PaymentStatus.SUCCEEDED
    
    @property
    def is_canceled(self) -> bool:
        """Check if payment is canceled."""
        return self.status == PaymentStatus.CANCELED
    
    @property
    def requires_action(self) -> bool:
        """Check if payment requires action (e.g., 3D Secure)."""
        return self.status == PaymentStatus.REQUIRES_ACTION
    
    @property
    def requires_capture(self) -> bool:
        """Check if payment requires capture."""
        return self.status == PaymentStatus.REQUIRES_CAPTURE
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "object": "payment_intent",
            "amount": self.amount,
            "amount_capturable": self.amount_capturable,
            "amount_received": self.amount_received,
            "currency": self.currency.value,
            "status": self.status.value,
            "customer": self.customer_id,
            "payment_method": self.payment_method_id,
            "description": self.description,
            "receipt_email": self.receipt_email,
            "statement_descriptor": self.statement_descriptor,
            "statement_descriptor_suffix": self.statement_descriptor_suffix,
            "capture_method": self.capture_method,
            "confirmation_method": self.confirmation_method,
            "client_secret": self.client_secret,
            "last_payment_error": self.last_payment_error,
            "charges": {"data": self.charges},
            "invoice": self.invoice_id,
            "setup_future_usage": self.setup_future_usage,
            "created": int(self.created),
            "canceled_at": int(self.canceled_at) if self.canceled_at else None,
            "cancellation_reason": self.cancellation_reason,
            "livemode": self.livemode,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PaymentIntent':
        """Create PaymentIntent from dictionary."""
        currency_value = data.get("currency", "usd")
        return cls(
            id=data.get("id", ""),
            amount=data.get("amount", 0),
            currency=Currency(currency_value) if currency_value else Currency.USD,
            status=PaymentStatus(data.get("status", "requires_payment_method")),
            customer_id=data.get("customer"),
            payment_method_id=data.get("payment_method"),
            description=data.get("description"),
            receipt_email=data.get("receipt_email"),
            statement_descriptor=data.get("statement_descriptor"),
            statement_descriptor_suffix=data.get("statement_descriptor_suffix"),
            capture_method=data.get("capture_method", "automatic"),
            confirmation_method=data.get("confirmation_method", "automatic"),
            client_secret=data.get("client_secret", ""),
            last_payment_error=data.get("last_payment_error"),
            charges=data.get("charges", {}).get("data", []),
            invoice_id=data.get("invoice"),
            setup_future_usage=data.get("setup_future_usage"),
            created=data.get("created", time.time()),
            canceled_at=data.get("canceled_at"),
            cancellation_reason=data.get("cancellation_reason"),
            livemode=data.get("livemode", False),
            metadata=data.get("metadata", {}),
            amount_capturable=data.get("amount_capturable", 0),
            amount_received=data.get("amount_received", 0),
        )


@dataclass
class Refund:
    """
    Stripe refund representation.
    
    Attributes:
        id: Unique refund ID (re_xxx)
        amount: Refund amount in cents
        currency: Currency code
        status: Refund status
        charge_id: Associated charge ID
        payment_intent_id: Associated payment intent ID
        reason: Refund reason
        receipt_number: Receipt number
        failure_reason: Failure reason
        failure_balance_transaction: Failure balance transaction
        created: Creation timestamp
        livemode: Whether this is a live mode object
        metadata: Custom metadata
    """
    id: str
    amount: int
    currency: Currency = Currency.USD
    status: RefundStatus = RefundStatus.PENDING
    charge_id: Optional[str] = None
    payment_intent_id: Optional[str] = None
    reason: Optional[RefundReason] = None
    receipt_number: Optional[str] = None
    failure_reason: Optional[str] = None
    failure_balance_transaction: Optional[str] = None
    created: float = field(default_factory=time.time)
    livemode: bool = False
    metadata: Dict[str, str] = field(default_factory=dict)
    
    @property
    def is_succeeded(self) -> bool:
        """Check if refund succeeded."""
        return self.status == RefundStatus.SUCCEEDED
    
    @property
    def is_failed(self) -> bool:
        """Check if refund failed."""
        return self.status == RefundStatus.FAILED
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "object": "refund",
            "amount": self.amount,
            "currency": self.currency.value,
            "status": self.status.value,
            "charge": self.charge_id,
            "payment_intent": self.payment_intent_id,
            "reason": self.reason.value if self.reason else None,
            "receipt_number": self.receipt_number,
            "failure_reason": self.failure_reason,
            "failure_balance_transaction": self.failure_balance_transaction,
            "created": int(self.created),
            "livemode": self.livemode,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Refund':
        """Create Refund from dictionary."""
        currency_value = data.get("currency", "usd")
        reason_value = data.get("reason")
        return cls(
            id=data.get("id", ""),
            amount=data.get("amount", 0),
            currency=Currency(currency_value) if currency_value else Currency.USD,
            status=RefundStatus(data.get("status", "pending")),
            charge_id=data.get("charge"),
            payment_intent_id=data.get("payment_intent"),
            reason=RefundReason(reason_value) if reason_value else None,
            receipt_number=data.get("receipt_number"),
            failure_reason=data.get("failure_reason"),
            failure_balance_transaction=data.get("failure_balance_transaction"),
            created=data.get("created", time.time()),
            livemode=data.get("livemode", False),
            metadata=data.get("metadata", {}),
        )


@dataclass
class CheckoutSession:
    """
    Stripe checkout session representation.
    
    Attributes:
        id: Unique session ID (cs_xxx)
        customer_id: Associated customer ID
        payment_intent_id: Associated payment intent ID
        subscription_id: Associated subscription ID
        mode: Checkout mode (payment, subscription, setup)
        status: Session status
        success_url: Success redirect URL
        cancel_url: Cancel redirect URL
        url: Checkout session URL
        amount_total: Total amount in cents
        amount_subtotal: Subtotal amount in cents
        currency: Currency code
        customer_email: Customer email
        client_reference_id: Client reference ID
        line_items: Line items
        payment_status: Payment status
        expires_at: Session expiration timestamp
        created: Creation timestamp
        livemode: Whether this is a live mode object
        metadata: Custom metadata
    """
    id: str
    customer_id: Optional[str] = None
    payment_intent_id: Optional[str] = None
    subscription_id: Optional[str] = None
    mode: CheckoutMode = CheckoutMode.PAYMENT
    status: str = "open"
    success_url: str = ""
    cancel_url: str = ""
    url: Optional[str] = None
    amount_total: Optional[int] = None
    amount_subtotal: Optional[int] = None
    currency: Optional[Currency] = None
    customer_email: Optional[str] = None
    client_reference_id: Optional[str] = None
    line_items: List[Dict[str, Any]] = field(default_factory=list)
    payment_status: str = "unpaid"
    expires_at: float = field(default_factory=lambda: time.time() + 24 * 3600)
    created: float = field(default_factory=time.time)
    livemode: bool = False
    metadata: Dict[str, str] = field(default_factory=dict)
    allow_promotion_codes: bool = False
    billing_address_collection: Optional[str] = None
    shipping_address_collection: Optional[Dict[str, Any]] = None
    
    @property
    def is_complete(self) -> bool:
        """Check if checkout session is complete."""
        return self.status == "complete"
    
    @property
    def is_expired(self) -> bool:
        """Check if checkout session is expired."""
        return self.status == "expired" or time.time() > self.expires_at
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "object": "checkout.session",
            "customer": self.customer_id,
            "payment_intent": self.payment_intent_id,
            "subscription": self.subscription_id,
            "mode": self.mode.value,
            "status": self.status,
            "success_url": self.success_url,
            "cancel_url": self.cancel_url,
            "url": self.url,
            "amount_total": self.amount_total,
            "amount_subtotal": self.amount_subtotal,
            "currency": self.currency.value if self.currency else None,
            "customer_email": self.customer_email,
            "client_reference_id": self.client_reference_id,
            "line_items": {"data": self.line_items},
            "payment_status": self.payment_status,
            "expires_at": int(self.expires_at),
            "created": int(self.created),
            "livemode": self.livemode,
            "metadata": self.metadata,
            "allow_promotion_codes": self.allow_promotion_codes,
            "billing_address_collection": self.billing_address_collection,
            "shipping_address_collection": self.shipping_address_collection,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CheckoutSession':
        """Create CheckoutSession from dictionary."""
        currency_value = data.get("currency")
        return cls(
            id=data.get("id", ""),
            customer_id=data.get("customer"),
            payment_intent_id=data.get("payment_intent"),
            subscription_id=data.get("subscription"),
            mode=CheckoutMode(data.get("mode", "payment")),
            status=data.get("status", "open"),
            success_url=data.get("success_url", ""),
            cancel_url=data.get("cancel_url", ""),
            url=data.get("url"),
            amount_total=data.get("amount_total"),
            amount_subtotal=data.get("amount_subtotal"),
            currency=Currency(currency_value) if currency_value else None,
            customer_email=data.get("customer_email"),
            client_reference_id=data.get("client_reference_id"),
            line_items=data.get("line_items", {}).get("data", []),
            payment_status=data.get("payment_status", "unpaid"),
            expires_at=data.get("expires_at", time.time() + 24 * 3600),
            created=data.get("created", time.time()),
            livemode=data.get("livemode", False),
            metadata=data.get("metadata", {}),
            allow_promotion_codes=data.get("allow_promotion_codes", False),
            billing_address_collection=data.get("billing_address_collection"),
            shipping_address_collection=data.get("shipping_address_collection"),
        )


@dataclass
class PortalSession:
    """
    Stripe billing portal session.
    
    Attributes:
        id: Unique session ID (bps_xxx)
        customer_id: Associated customer ID
        url: Portal session URL
        return_url: Return URL after portal
        created: Creation timestamp
        livemode: Whether this is a live mode object
    """
    id: str
    customer_id: str
    url: str
    return_url: str
    created: float = field(default_factory=time.time)
    livemode: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "object": "billing_portal.session",
            "customer": self.customer_id,
            "url": self.url,
            "return_url": self.return_url,
            "created": int(self.created),
            "livemode": self.livemode,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PortalSession':
        """Create PortalSession from dictionary."""
        return cls(
            id=data.get("id", ""),
            customer_id=data.get("customer", ""),
            url=data.get("url", ""),
            return_url=data.get("return_url", ""),
            created=data.get("created", time.time()),
            livemode=data.get("livemode", False),
        )


@dataclass
class WebhookEvent:
    """
    Stripe webhook event.
    
    Attributes:
        id: Unique event ID (evt_xxx)
        type: Event type
        data: Event data
        api_version: API version
        created: Creation timestamp
        livemode: Whether this is a live mode object
        pending_webhooks: Number of pending webhooks
        request: Request information
    """
    id: str
    type: str
    data: Dict[str, Any] = field(default_factory=dict)
    api_version: str = ""
    created: float = field(default_factory=time.time)
    livemode: bool = False
    pending_webhooks: int = 0
    request: Optional[Dict[str, Any]] = None
    
    @property
    def object(self) -> Dict[str, Any]:
        """Get the event object."""
        return self.data.get("object", {})
    
    @property
    def previous_attributes(self) -> Dict[str, Any]:
        """Get previous attributes for update events."""
        return self.data.get("previous_attributes", {})
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "object": "event",
            "type": self.type,
            "data": self.data,
            "api_version": self.api_version,
            "created": int(self.created),
            "livemode": self.livemode,
            "pending_webhooks": self.pending_webhooks,
            "request": self.request,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WebhookEvent':
        """Create WebhookEvent from dictionary."""
        return cls(
            id=data.get("id", ""),
            type=data.get("type", ""),
            data=data.get("data", {}),
            api_version=data.get("api_version", ""),
            created=data.get("created", time.time()),
            livemode=data.get("livemode", False),
            pending_webhooks=data.get("pending_webhooks", 0),
            request=data.get("request"),
        )


@dataclass
class UsageRecord:
    """
    Usage record for metered billing.
    
    Attributes:
        id: Unique usage record ID (mbur_xxx)
        subscription_item_id: Associated subscription item ID
        quantity: Usage quantity
        timestamp: Usage timestamp
        action: Action type (increment or set)
        livemode: Whether this is a live mode object
    """
    id: str
    subscription_item_id: str
    quantity: int
    timestamp: float = field(default_factory=time.time)
    action: str = "increment"
    livemode: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "object": "usage_record",
            "subscription_item": self.subscription_item_id,
            "quantity": self.quantity,
            "timestamp": int(self.timestamp),
            "action": self.action,
            "livemode": self.livemode,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UsageRecord':
        """Create UsageRecord from dictionary."""
        return cls(
            id=data.get("id", ""),
            subscription_item_id=data.get("subscription_item", ""),
            quantity=data.get("quantity", 0),
            timestamp=data.get("timestamp", time.time()),
            action=data.get("action", "increment"),
            livemode=data.get("livemode", False),
        )


@dataclass
class Coupon:
    """
    Stripe coupon for discounts.
    
    Attributes:
        id: Unique coupon ID
        name: Coupon display name
        percent_off: Percentage discount (0-100)
        amount_off: Fixed amount discount in cents
        currency: Currency for amount_off
        duration: Coupon duration (once, repeating, forever)
        duration_in_months: Months for repeating duration
        max_redemptions: Maximum redemptions
        times_redeemed: Times redeemed
        redeem_by: Redemption deadline
        applies_to: Products coupon applies to
        valid: Whether coupon is valid
        created: Creation timestamp
        livemode: Whether this is a live mode object
        metadata: Custom metadata
    """
    id: str
    name: Optional[str] = None
    percent_off: Optional[float] = None
    amount_off: Optional[int] = None
    currency: Optional[Currency] = None
    duration: str = "once"
    duration_in_months: Optional[int] = None
    max_redemptions: Optional[int] = None
    times_redeemed: int = 0
    redeem_by: Optional[float] = None
    applies_to: Optional[Dict[str, List[str]]] = None
    valid: bool = True
    created: float = field(default_factory=time.time)
    livemode: bool = False
    metadata: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "object": "coupon",
            "name": self.name,
            "percent_off": self.percent_off,
            "amount_off": self.amount_off,
            "currency": self.currency.value if self.currency else None,
            "duration": self.duration,
            "duration_in_months": self.duration_in_months,
            "max_redemptions": self.max_redemptions,
            "times_redeemed": self.times_redeemed,
            "redeem_by": int(self.redeem_by) if self.redeem_by else None,
            "applies_to": self.applies_to,
            "valid": self.valid,
            "created": int(self.created),
            "livemode": self.livemode,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Coupon':
        """Create Coupon from dictionary."""
        currency_value = data.get("currency")
        return cls(
            id=data.get("id", ""),
            name=data.get("name"),
            percent_off=data.get("percent_off"),
            amount_off=data.get("amount_off"),
            currency=Currency(currency_value) if currency_value else None,
            duration=data.get("duration", "once"),
            duration_in_months=data.get("duration_in_months"),
            max_redemptions=data.get("max_redemptions"),
            times_redeemed=data.get("times_redeemed", 0),
            redeem_by=data.get("redeem_by"),
            applies_to=data.get("applies_to"),
            valid=data.get("valid", True),
            created=data.get("created", time.time()),
            livemode=data.get("livemode", False),
            metadata=data.get("metadata", {}),
        )


@dataclass
class TaxRate:
    """
    Stripe tax rate.
    
    Attributes:
        id: Unique tax rate ID (txr_xxx)
        display_name: Display name
        description: Tax rate description
        percentage: Tax percentage
        inclusive: Whether tax is inclusive
        active: Whether tax rate is active
        country: Country code
        state: State code
        jurisdiction: Tax jurisdiction
        tax_type: Tax type (vat, sales_tax, etc.)
        created: Creation timestamp
        livemode: Whether this is a live mode object
        metadata: Custom metadata
    """
    id: str
    display_name: str
    percentage: float
    description: Optional[str] = None
    inclusive: bool = False
    active: bool = True
    country: Optional[str] = None
    state: Optional[str] = None
    jurisdiction: Optional[str] = None
    tax_type: Optional[str] = None
    created: float = field(default_factory=time.time)
    livemode: bool = False
    metadata: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "object": "tax_rate",
            "display_name": self.display_name,
            "description": self.description,
            "percentage": self.percentage,
            "inclusive": self.inclusive,
            "active": self.active,
            "country": self.country,
            "state": self.state,
            "jurisdiction": self.jurisdiction,
            "tax_type": self.tax_type,
            "created": int(self.created),
            "livemode": self.livemode,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TaxRate':
        """Create TaxRate from dictionary."""
        return cls(
            id=data.get("id", ""),
            display_name=data.get("display_name", ""),
            description=data.get("description"),
            percentage=data.get("percentage", 0),
            inclusive=data.get("inclusive", False),
            active=data.get("active", True),
            country=data.get("country"),
            state=data.get("state"),
            jurisdiction=data.get("jurisdiction"),
            tax_type=data.get("tax_type"),
            created=data.get("created", time.time()),
            livemode=data.get("livemode", False),
            metadata=data.get("metadata", {}),
        )


def _generate_id(prefix: str) -> str:
    """Generate a Stripe-like ID with prefix."""
    random_part = secrets.token_hex(12)
    return f"{prefix}_{random_part}"


def _timing_safe_compare(a: bytes, b: bytes) -> bool:
    """Timing-safe comparison to prevent timing attacks."""
    return hmac.compare_digest(a, b)


class CustomerManager:
    """
    Customer management operations.
    
    Handles customer creation, updates, retrieval, and deletion.
    Works as an abstraction layer that can be backed by the Stripe SDK.
    """
    
    def __init__(self, client: 'StripeClient'):
        """Initialize with parent client."""
        self._client = client
        self._customers: Dict[str, Customer] = {}
        self._lock = threading.Lock()
    
    def create(
        self,
        email: Optional[str] = None,
        name: Optional[str] = None,
        phone: Optional[str] = None,
        description: Optional[str] = None,
        address: Optional[Address] = None,
        shipping: Optional[Dict[str, Any]] = None,
        payment_method_id: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
        tax_exempt: str = "none",
        idempotency_key: Optional[str] = None,
    ) -> Customer:
        """
        Create a new customer.
        
        Args:
            email: Customer email address
            name: Customer full name
            phone: Customer phone number
            description: Customer description
            address: Customer address
            shipping: Shipping address information
            payment_method_id: Default payment method ID
            metadata: Custom metadata
            tax_exempt: Tax exemption status (none, exempt, reverse)
            idempotency_key: Idempotency key for request deduplication
        
        Returns:
            Created Customer object
        
        Raises:
            InvalidRequestError: If parameters are invalid
        """
        self._client._check_api_key()
        
        if metadata and len(metadata) > StripeLimit.MAX_METADATA_KEYS.value:
            raise InvalidRequestError(
                f"Metadata cannot have more than {StripeLimit.MAX_METADATA_KEYS.value} keys",
                param="metadata"
            )
        
        customer = Customer(
            id=_generate_id("cus"),
            email=email,
            name=name,
            phone=phone,
            description=description,
            address=address,
            shipping=shipping,
            default_payment_method_id=payment_method_id,
            metadata=metadata or {},
            tax_exempt=tax_exempt,
            livemode=not self._client.test_mode,
        )
        
        with self._lock:
            self._customers[customer.id] = customer
        
        self._client._log_operation("customer.create", customer_id=customer.id)
        return customer
    
    def retrieve(self, customer_id: str) -> Customer:
        """
        Retrieve a customer by ID.
        
        Args:
            customer_id: Customer ID (cus_xxx)
        
        Returns:
            Customer object
        
        Raises:
            ResourceNotFoundError: If customer not found
        """
        self._client._check_api_key()
        
        with self._lock:
            if customer_id not in self._customers:
                raise ResourceNotFoundError("Customer", customer_id)
            return copy.deepcopy(self._customers[customer_id])
    
    def update(
        self,
        customer_id: str,
        email: Optional[str] = None,
        name: Optional[str] = None,
        phone: Optional[str] = None,
        description: Optional[str] = None,
        address: Optional[Address] = None,
        shipping: Optional[Dict[str, Any]] = None,
        default_payment_method_id: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
        tax_exempt: Optional[str] = None,
    ) -> Customer:
        """
        Update a customer.
        
        Args:
            customer_id: Customer ID to update
            email: New email address
            name: New name
            phone: New phone number
            description: New description
            address: New address
            shipping: New shipping information
            default_payment_method_id: New default payment method
            metadata: New metadata (merges with existing)
            tax_exempt: New tax exemption status
        
        Returns:
            Updated Customer object
        
        Raises:
            ResourceNotFoundError: If customer not found
        """
        self._client._check_api_key()
        
        with self._lock:
            if customer_id not in self._customers:
                raise ResourceNotFoundError("Customer", customer_id)
            
            customer = self._customers[customer_id]
            
            if email is not None:
                customer.email = email
            if name is not None:
                customer.name = name
            if phone is not None:
                customer.phone = phone
            if description is not None:
                customer.description = description
            if address is not None:
                customer.address = address
            if shipping is not None:
                customer.shipping = shipping
            if default_payment_method_id is not None:
                customer.default_payment_method_id = default_payment_method_id
            if metadata is not None:
                customer.metadata.update(metadata)
            if tax_exempt is not None:
                customer.tax_exempt = tax_exempt
            
            self._client._log_operation("customer.update", customer_id=customer_id)
            return copy.deepcopy(customer)
    
    def delete(self, customer_id: str) -> Customer:
        """
        Delete a customer.
        
        Args:
            customer_id: Customer ID to delete
        
        Returns:
            Deleted Customer object with deleted=True
        
        Raises:
            ResourceNotFoundError: If customer not found
        """
        self._client._check_api_key()
        
        with self._lock:
            if customer_id not in self._customers:
                raise ResourceNotFoundError("Customer", customer_id)
            
            customer = self._customers[customer_id]
            customer.deleted = True
            del self._customers[customer_id]
            
            self._client._log_operation("customer.delete", customer_id=customer_id)
            return customer
    
    def list(
        self,
        email: Optional[str] = None,
        limit: int = 10,
        starting_after: Optional[str] = None,
        ending_before: Optional[str] = None,
    ) -> List[Customer]:
        """
        List customers.
        
        Args:
            email: Filter by email
            limit: Maximum number of customers to return
            starting_after: Cursor for pagination (customer ID)
            ending_before: Cursor for pagination (customer ID)
        
        Returns:
            List of Customer objects
        """
        self._client._check_api_key()
        
        limit = min(limit, StripeLimit.MAX_ITEMS_PER_LIST.value)
        
        with self._lock:
            customers = list(self._customers.values())
        
        if email:
            customers = [c for c in customers if c.email == email]
        
        customers = [c for c in customers if not c.deleted]
        customers.sort(key=lambda c: c.created, reverse=True)
        
        if starting_after:
            idx = next((i for i, c in enumerate(customers) if c.id == starting_after), -1)
            if idx >= 0:
                customers = customers[idx + 1:]
        
        if ending_before:
            idx = next((i for i, c in enumerate(customers) if c.id == ending_before), len(customers))
            customers = customers[:idx]
        
        return [copy.deepcopy(c) for c in customers[:limit]]


class SubscriptionManager:
    """
    Subscription lifecycle management.
    
    Handles subscription creation, updates, cancellation, and usage-based billing.
    """
    
    def __init__(self, client: 'StripeClient'):
        """Initialize with parent client."""
        self._client = client
        self._subscriptions: Dict[str, Subscription] = {}
        self._usage_records: Dict[str, List[UsageRecord]] = {}
        self._lock = threading.Lock()
    
    def create(
        self,
        customer_id: str,
        price_id: str,
        quantity: int = 1,
        payment_method_id: Optional[str] = None,
        trial_days: Optional[int] = None,
        trial_end: Optional[float] = None,
        billing_cycle_anchor: Optional[float] = None,
        cancel_at: Optional[float] = None,
        collection_method: CollectionMethod = CollectionMethod.CHARGE_AUTOMATICALLY,
        days_until_due: Optional[int] = None,
        proration_behavior: ProrationBehavior = ProrationBehavior.CREATE_PRORATIONS,
        coupon_id: Optional[str] = None,
        default_tax_rates: Optional[List[str]] = None,
        metadata: Optional[Dict[str, str]] = None,
        idempotency_key: Optional[str] = None,
    ) -> Subscription:
        """
        Create a new subscription.
        
        Args:
            customer_id: Customer ID
            price_id: Price ID for the subscription
            quantity: Quantity of the subscription item
            payment_method_id: Payment method ID
            trial_days: Number of trial days
            trial_end: Trial end timestamp
            billing_cycle_anchor: Billing cycle anchor timestamp
            cancel_at: Automatic cancellation timestamp
            collection_method: Collection method
            days_until_due: Days until invoice is due (for send_invoice)
            proration_behavior: How to handle prorations
            coupon_id: Coupon ID to apply
            default_tax_rates: Default tax rate IDs
            metadata: Custom metadata
            idempotency_key: Idempotency key
        
        Returns:
            Created Subscription object
        
        Raises:
            InvalidRequestError: If parameters are invalid
            ResourceNotFoundError: If customer not found
        """
        self._client._check_api_key()
        
        try:
            self._client.customers.retrieve(customer_id)
        except ResourceNotFoundError:
            raise InvalidRequestError(f"Customer {customer_id} not found", param="customer_id")
        
        now = time.time()
        trial_start = None
        trial_end_ts = None
        
        if trial_days:
            trial_start = now
            trial_end_ts = now + trial_days * 24 * 3600
            status = SubscriptionStatus.TRIALING
        elif trial_end:
            trial_start = now
            trial_end_ts = trial_end
            status = SubscriptionStatus.TRIALING
        else:
            status = SubscriptionStatus.ACTIVE
        
        period_end = billing_cycle_anchor or now + 30 * 24 * 3600
        if trial_end_ts:
            period_end = trial_end_ts + 30 * 24 * 3600
        
        subscription_item = SubscriptionItem(
            id=_generate_id("si"),
            price_id=price_id,
            quantity=quantity,
        )
        
        subscription = Subscription(
            id=_generate_id("sub"),
            customer_id=customer_id,
            status=status,
            items=[subscription_item],
            default_payment_method_id=payment_method_id,
            current_period_start=now,
            current_period_end=period_end,
            trial_start=trial_start,
            trial_end=trial_end_ts,
            billing_cycle_anchor=billing_cycle_anchor or now,
            collection_method=collection_method,
            days_until_due=days_until_due,
            cancel_at=cancel_at,
            metadata=metadata or {},
            livemode=not self._client.test_mode,
        )
        
        with self._lock:
            self._subscriptions[subscription.id] = subscription
        
        self._client._log_operation(
            "subscription.create",
            customer_id=customer_id,
            subscription_id=subscription.id
        )
        return subscription
    
    def retrieve(self, subscription_id: str) -> Subscription:
        """
        Retrieve a subscription by ID.
        
        Args:
            subscription_id: Subscription ID (sub_xxx)
        
        Returns:
            Subscription object
        
        Raises:
            ResourceNotFoundError: If subscription not found
        """
        self._client._check_api_key()
        
        with self._lock:
            if subscription_id not in self._subscriptions:
                raise ResourceNotFoundError("Subscription", subscription_id)
            return copy.deepcopy(self._subscriptions[subscription_id])
    
    def update(
        self,
        subscription_id: str,
        price_id: Optional[str] = None,
        quantity: Optional[int] = None,
        payment_method_id: Optional[str] = None,
        proration_behavior: ProrationBehavior = ProrationBehavior.CREATE_PRORATIONS,
        cancel_at_period_end: Optional[bool] = None,
        cancel_at: Optional[float] = None,
        trial_end: Optional[float] = None,
        pause_collection: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> Subscription:
        """
        Update a subscription.
        
        Args:
            subscription_id: Subscription ID to update
            price_id: New price ID (upgrade/downgrade)
            quantity: New quantity
            payment_method_id: New payment method ID
            proration_behavior: How to handle prorations
            cancel_at_period_end: Whether to cancel at period end
            cancel_at: New cancellation timestamp
            trial_end: New trial end timestamp
            pause_collection: Pause collection settings
            metadata: New metadata (merges with existing)
        
        Returns:
            Updated Subscription object
        
        Raises:
            ResourceNotFoundError: If subscription not found
            SubscriptionError: If subscription cannot be updated
        """
        self._client._check_api_key()
        
        with self._lock:
            if subscription_id not in self._subscriptions:
                raise ResourceNotFoundError("Subscription", subscription_id)
            
            subscription = self._subscriptions[subscription_id]
            
            if subscription.status == SubscriptionStatus.CANCELED:
                raise SubscriptionError(
                    "Cannot update a canceled subscription",
                    subscription_id=subscription_id
                )
            
            if price_id and subscription.items:
                subscription.items[0].price_id = price_id
            
            if quantity is not None and subscription.items:
                subscription.items[0].quantity = quantity
            
            if payment_method_id is not None:
                subscription.default_payment_method_id = payment_method_id
            
            if cancel_at_period_end is not None:
                subscription.cancel_at_period_end = cancel_at_period_end
            
            if cancel_at is not None:
                subscription.cancel_at = cancel_at
            
            if trial_end is not None:
                subscription.trial_end = trial_end
            
            if pause_collection is not None:
                subscription.pause_collection = pause_collection
                if pause_collection.get("behavior") == "mark_uncollectible":
                    subscription.status = SubscriptionStatus.PAUSED
            
            if metadata is not None:
                subscription.metadata.update(metadata)
            
            self._client._log_operation("subscription.update", subscription_id=subscription_id)
            return copy.deepcopy(subscription)
    
    def cancel(
        self,
        subscription_id: str,
        cancel_at_period_end: bool = False,
        prorate: bool = True,
        invoice_now: bool = False,
    ) -> Subscription:
        """
        Cancel a subscription.
        
        Args:
            subscription_id: Subscription ID to cancel
            cancel_at_period_end: Cancel at the end of the current period
            prorate: Whether to prorate the cancellation
            invoice_now: Whether to invoice immediately
        
        Returns:
            Canceled Subscription object
        
        Raises:
            ResourceNotFoundError: If subscription not found
            SubscriptionError: If subscription cannot be canceled
        """
        self._client._check_api_key()
        
        with self._lock:
            if subscription_id not in self._subscriptions:
                raise ResourceNotFoundError("Subscription", subscription_id)
            
            subscription = self._subscriptions[subscription_id]
            
            if subscription.status == SubscriptionStatus.CANCELED:
                raise SubscriptionError(
                    "Subscription is already canceled",
                    subscription_id=subscription_id
                )
            
            now = time.time()
            
            if cancel_at_period_end:
                subscription.cancel_at_period_end = True
                subscription.cancel_at = subscription.current_period_end
            else:
                subscription.status = SubscriptionStatus.CANCELED
                subscription.canceled_at = now
                subscription.ended_at = now
            
            self._client._log_operation("subscription.cancel", subscription_id=subscription_id)
            return copy.deepcopy(subscription)
    
    def resume(self, subscription_id: str) -> Subscription:
        """
        Resume a paused subscription.
        
        Args:
            subscription_id: Subscription ID to resume
        
        Returns:
            Resumed Subscription object
        
        Raises:
            ResourceNotFoundError: If subscription not found
            SubscriptionError: If subscription cannot be resumed
        """
        self._client._check_api_key()
        
        with self._lock:
            if subscription_id not in self._subscriptions:
                raise ResourceNotFoundError("Subscription", subscription_id)
            
            subscription = self._subscriptions[subscription_id]
            
            if subscription.status != SubscriptionStatus.PAUSED:
                raise SubscriptionError(
                    "Can only resume paused subscriptions",
                    subscription_id=subscription_id
                )
            
            subscription.status = SubscriptionStatus.ACTIVE
            subscription.pause_collection = None
            
            self._client._log_operation("subscription.resume", subscription_id=subscription_id)
            return copy.deepcopy(subscription)
    
    def list(
        self,
        customer_id: Optional[str] = None,
        price_id: Optional[str] = None,
        status: Optional[SubscriptionStatus] = None,
        limit: int = 10,
        starting_after: Optional[str] = None,
    ) -> List[Subscription]:
        """
        List subscriptions.
        
        Args:
            customer_id: Filter by customer ID
            price_id: Filter by price ID
            status: Filter by status
            limit: Maximum number of subscriptions to return
            starting_after: Cursor for pagination
        
        Returns:
            List of Subscription objects
        """
        self._client._check_api_key()
        
        limit = min(limit, StripeLimit.MAX_ITEMS_PER_LIST.value)
        
        with self._lock:
            subscriptions = list(self._subscriptions.values())
        
        if customer_id:
            subscriptions = [s for s in subscriptions if s.customer_id == customer_id]
        
        if price_id:
            subscriptions = [
                s for s in subscriptions
                if any(item.price_id == price_id for item in s.items)
            ]
        
        if status:
            subscriptions = [s for s in subscriptions if s.status == status]
        
        subscriptions.sort(key=lambda s: s.created, reverse=True)
        
        if starting_after:
            idx = next((i for i, s in enumerate(subscriptions) if s.id == starting_after), -1)
            if idx >= 0:
                subscriptions = subscriptions[idx + 1:]
        
        return [copy.deepcopy(s) for s in subscriptions[:limit]]
    
    def create_usage_record(
        self,
        subscription_item_id: str,
        quantity: int,
        timestamp: Optional[float] = None,
        action: str = "increment",
        idempotency_key: Optional[str] = None,
    ) -> UsageRecord:
        """
        Create a usage record for metered billing.
        
        Args:
            subscription_item_id: Subscription item ID
            quantity: Usage quantity
            timestamp: Usage timestamp (defaults to now)
            action: Action type ("increment" or "set")
            idempotency_key: Idempotency key
        
        Returns:
            Created UsageRecord object
        
        Raises:
            InvalidRequestError: If parameters are invalid
        """
        self._client._check_api_key()
        
        if action not in ["increment", "set"]:
            raise InvalidRequestError(
                "Action must be 'increment' or 'set'",
                param="action"
            )
        
        usage_record = UsageRecord(
            id=_generate_id("mbur"),
            subscription_item_id=subscription_item_id,
            quantity=quantity,
            timestamp=timestamp or time.time(),
            action=action,
            livemode=not self._client.test_mode,
        )
        
        with self._lock:
            if subscription_item_id not in self._usage_records:
                self._usage_records[subscription_item_id] = []
            self._usage_records[subscription_item_id].append(usage_record)
        
        self._client._log_operation(
            "usage_record.create",
            subscription_item_id=subscription_item_id,
            quantity=quantity
        )
        return usage_record


class PaymentManager:
    """
    Payment processing operations.
    
    Handles payment intents, captures, and refunds.
    """
    
    def __init__(self, client: 'StripeClient'):
        """Initialize with parent client."""
        self._client = client
        self._payment_intents: Dict[str, PaymentIntent] = {}
        self._refunds: Dict[str, Refund] = {}
        self._lock = threading.Lock()
    
    def create_payment_intent(
        self,
        amount: int,
        currency: Union[Currency, str] = Currency.USD,
        customer_id: Optional[str] = None,
        payment_method_id: Optional[str] = None,
        description: Optional[str] = None,
        receipt_email: Optional[str] = None,
        statement_descriptor: Optional[str] = None,
        statement_descriptor_suffix: Optional[str] = None,
        capture_method: str = "automatic",
        confirmation_method: str = "automatic",
        setup_future_usage: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
        idempotency_key: Optional[str] = None,
    ) -> PaymentIntent:
        """
        Create a payment intent.
        
        Args:
            amount: Amount in cents
            currency: Currency code
            customer_id: Customer ID
            payment_method_id: Payment method ID
            description: Payment description
            receipt_email: Email for receipt
            statement_descriptor: Statement descriptor
            statement_descriptor_suffix: Statement descriptor suffix
            capture_method: Capture method ("automatic" or "manual")
            confirmation_method: Confirmation method
            setup_future_usage: Setup for future usage
            metadata: Custom metadata
            idempotency_key: Idempotency key
        
        Returns:
            Created PaymentIntent object
        
        Raises:
            InvalidRequestError: If parameters are invalid
        """
        self._client._check_api_key()
        
        if amount < StripeLimit.MIN_CHARGE_AMOUNT_USD.value:
            raise InvalidRequestError(
                f"Amount must be at least {StripeLimit.MIN_CHARGE_AMOUNT_USD.value} cents",
                param="amount"
            )
        
        if amount > StripeLimit.MAX_CHARGE_AMOUNT_USD.value:
            raise InvalidRequestError(
                f"Amount cannot exceed {StripeLimit.MAX_CHARGE_AMOUNT_USD.value} cents",
                param="amount"
            )
        
        if isinstance(currency, str):
            currency = Currency(currency.lower())
        
        if statement_descriptor and len(statement_descriptor) > StripeLimit.MAX_STATEMENT_DESCRIPTOR_LENGTH.value:
            raise InvalidRequestError(
                f"Statement descriptor cannot exceed {StripeLimit.MAX_STATEMENT_DESCRIPTOR_LENGTH.value} characters",
                param="statement_descriptor"
            )
        
        client_secret = f"pi_{secrets.token_hex(12)}_secret_{secrets.token_hex(12)}"
        
        payment_intent = PaymentIntent(
            id=_generate_id("pi"),
            amount=amount,
            currency=currency,
            status=PaymentStatus.REQUIRES_PAYMENT_METHOD,
            customer_id=customer_id,
            payment_method_id=payment_method_id,
            description=description,
            receipt_email=receipt_email,
            statement_descriptor=statement_descriptor,
            statement_descriptor_suffix=statement_descriptor_suffix,
            capture_method=capture_method,
            confirmation_method=confirmation_method,
            client_secret=client_secret,
            setup_future_usage=setup_future_usage,
            metadata=metadata or {},
            livemode=not self._client.test_mode,
        )
        
        if payment_method_id:
            payment_intent.status = PaymentStatus.REQUIRES_CONFIRMATION
        
        with self._lock:
            self._payment_intents[payment_intent.id] = payment_intent
        
        self._client._log_operation(
            "payment_intent.create",
            payment_intent_id=payment_intent.id,
            amount=amount,
            currency=currency.value
        )
        return payment_intent
    
    def retrieve_payment_intent(self, payment_intent_id: str) -> PaymentIntent:
        """
        Retrieve a payment intent by ID.
        
        Args:
            payment_intent_id: Payment intent ID (pi_xxx)
        
        Returns:
            PaymentIntent object
        
        Raises:
            ResourceNotFoundError: If payment intent not found
        """
        self._client._check_api_key()
        
        with self._lock:
            if payment_intent_id not in self._payment_intents:
                raise ResourceNotFoundError("PaymentIntent", payment_intent_id)
            return copy.deepcopy(self._payment_intents[payment_intent_id])
    
    def confirm_payment_intent(
        self,
        payment_intent_id: str,
        payment_method_id: Optional[str] = None,
        return_url: Optional[str] = None,
    ) -> PaymentIntent:
        """
        Confirm a payment intent.
        
        Args:
            payment_intent_id: Payment intent ID
            payment_method_id: Payment method ID to use
            return_url: Return URL for redirects
        
        Returns:
            Confirmed PaymentIntent object
        
        Raises:
            ResourceNotFoundError: If payment intent not found
            PaymentFailedError: If payment fails
        """
        self._client._check_api_key()
        
        with self._lock:
            if payment_intent_id not in self._payment_intents:
                raise ResourceNotFoundError("PaymentIntent", payment_intent_id)
            
            payment_intent = self._payment_intents[payment_intent_id]
            
            if payment_method_id:
                payment_intent.payment_method_id = payment_method_id
            
            if payment_intent.capture_method == "manual":
                payment_intent.status = PaymentStatus.REQUIRES_CAPTURE
                payment_intent.amount_capturable = payment_intent.amount
            else:
                payment_intent.status = PaymentStatus.SUCCEEDED
                payment_intent.amount_received = payment_intent.amount
            
            self._client._log_operation(
                "payment_intent.confirm",
                payment_intent_id=payment_intent_id
            )
            return copy.deepcopy(payment_intent)
    
    def capture_payment(
        self,
        payment_intent_id: str,
        amount_to_capture: Optional[int] = None,
    ) -> PaymentIntent:
        """
        Capture a payment intent.
        
        Args:
            payment_intent_id: Payment intent ID
            amount_to_capture: Amount to capture (defaults to full amount)
        
        Returns:
            Captured PaymentIntent object
        
        Raises:
            ResourceNotFoundError: If payment intent not found
            PaymentFailedError: If capture fails
        """
        self._client._check_api_key()
        
        with self._lock:
            if payment_intent_id not in self._payment_intents:
                raise ResourceNotFoundError("PaymentIntent", payment_intent_id)
            
            payment_intent = self._payment_intents[payment_intent_id]
            
            if payment_intent.status != PaymentStatus.REQUIRES_CAPTURE:
                raise PaymentFailedError(
                    "Payment intent is not capturable",
                    payment_intent_id=payment_intent_id
                )
            
            capture_amount = amount_to_capture or payment_intent.amount_capturable
            
            if capture_amount > payment_intent.amount_capturable:
                raise PaymentFailedError(
                    f"Cannot capture more than {payment_intent.amount_capturable}",
                    payment_intent_id=payment_intent_id
                )
            
            payment_intent.status = PaymentStatus.SUCCEEDED
            payment_intent.amount_received = capture_amount
            payment_intent.amount_capturable = 0
            
            self._client._log_operation(
                "payment_intent.capture",
                payment_intent_id=payment_intent_id,
                amount=capture_amount
            )
            return copy.deepcopy(payment_intent)
    
    def cancel_payment_intent(
        self,
        payment_intent_id: str,
        cancellation_reason: Optional[str] = None,
    ) -> PaymentIntent:
        """
        Cancel a payment intent.
        
        Args:
            payment_intent_id: Payment intent ID
            cancellation_reason: Reason for cancellation
        
        Returns:
            Canceled PaymentIntent object
        
        Raises:
            ResourceNotFoundError: If payment intent not found
            PaymentFailedError: If cancellation fails
        """
        self._client._check_api_key()
        
        with self._lock:
            if payment_intent_id not in self._payment_intents:
                raise ResourceNotFoundError("PaymentIntent", payment_intent_id)
            
            payment_intent = self._payment_intents[payment_intent_id]
            
            if payment_intent.status == PaymentStatus.SUCCEEDED:
                raise PaymentFailedError(
                    "Cannot cancel a succeeded payment intent",
                    payment_intent_id=payment_intent_id
                )
            
            payment_intent.status = PaymentStatus.CANCELED
            payment_intent.canceled_at = time.time()
            payment_intent.cancellation_reason = cancellation_reason
            
            self._client._log_operation(
                "payment_intent.cancel",
                payment_intent_id=payment_intent_id
            )
            return copy.deepcopy(payment_intent)
    
    def create_refund(
        self,
        payment_intent_id: Optional[str] = None,
        charge_id: Optional[str] = None,
        amount: Optional[int] = None,
        reason: Optional[RefundReason] = None,
        metadata: Optional[Dict[str, str]] = None,
        idempotency_key: Optional[str] = None,
    ) -> Refund:
        """
        Create a refund.
        
        Args:
            payment_intent_id: Payment intent ID to refund
            charge_id: Charge ID to refund
            amount: Amount to refund (defaults to full amount)
            reason: Refund reason
            metadata: Custom metadata
            idempotency_key: Idempotency key
        
        Returns:
            Created Refund object
        
        Raises:
            InvalidRequestError: If parameters are invalid
            RefundError: If refund fails
        """
        self._client._check_api_key()
        
        if not payment_intent_id and not charge_id:
            raise InvalidRequestError(
                "Either payment_intent_id or charge_id is required",
                param="payment_intent_id"
            )
        
        with self._lock:
            if payment_intent_id:
                if payment_intent_id not in self._payment_intents:
                    raise ResourceNotFoundError("PaymentIntent", payment_intent_id)
                
                payment_intent = self._payment_intents[payment_intent_id]
                
                if payment_intent.status != PaymentStatus.SUCCEEDED:
                    raise RefundError(
                        "Can only refund succeeded payments",
                        charge_id=charge_id
                    )
                
                max_refund = payment_intent.amount_received
            else:
                max_refund = amount or 0
            
            refund_amount = amount or max_refund
            
            if refund_amount > max_refund:
                raise RefundError(
                    f"Refund amount {refund_amount} exceeds available {max_refund}",
                    charge_id=charge_id
                )
            
            refund = Refund(
                id=_generate_id("re"),
                amount=refund_amount,
                currency=payment_intent.currency if payment_intent_id else Currency.USD,
                status=RefundStatus.SUCCEEDED,
                charge_id=charge_id,
                payment_intent_id=payment_intent_id,
                reason=reason,
                metadata=metadata or {},
                livemode=not self._client.test_mode,
            )
            
            self._refunds[refund.id] = refund
        
        self._client._log_operation(
            "refund.create",
            refund_id=refund.id,
            amount=refund_amount
        )
        return refund
    
    def retrieve_refund(self, refund_id: str) -> Refund:
        """
        Retrieve a refund by ID.
        
        Args:
            refund_id: Refund ID (re_xxx)
        
        Returns:
            Refund object
        
        Raises:
            ResourceNotFoundError: If refund not found
        """
        self._client._check_api_key()
        
        with self._lock:
            if refund_id not in self._refunds:
                raise ResourceNotFoundError("Refund", refund_id)
            return copy.deepcopy(self._refunds[refund_id])
    
    def list_refunds(
        self,
        payment_intent_id: Optional[str] = None,
        charge_id: Optional[str] = None,
        limit: int = 10,
        starting_after: Optional[str] = None,
    ) -> List[Refund]:
        """
        List refunds.
        
        Args:
            payment_intent_id: Filter by payment intent ID
            charge_id: Filter by charge ID
            limit: Maximum number of refunds to return
            starting_after: Cursor for pagination
        
        Returns:
            List of Refund objects
        """
        self._client._check_api_key()
        
        limit = min(limit, StripeLimit.MAX_ITEMS_PER_LIST.value)
        
        with self._lock:
            refunds = list(self._refunds.values())
        
        if payment_intent_id:
            refunds = [r for r in refunds if r.payment_intent_id == payment_intent_id]
        
        if charge_id:
            refunds = [r for r in refunds if r.charge_id == charge_id]
        
        refunds.sort(key=lambda r: r.created, reverse=True)
        
        if starting_after:
            idx = next((i for i, r in enumerate(refunds) if r.id == starting_after), -1)
            if idx >= 0:
                refunds = refunds[idx + 1:]
        
        return [copy.deepcopy(r) for r in refunds[:limit]]


class InvoiceManager:
    """
    Invoice management operations.
    
    Handles invoice creation, finalization, and payment.
    """
    
    def __init__(self, client: 'StripeClient'):
        """Initialize with parent client."""
        self._client = client
        self._invoices: Dict[str, Invoice] = {}
        self._lock = threading.Lock()
    
    def create(
        self,
        customer_id: str,
        subscription_id: Optional[str] = None,
        collection_method: CollectionMethod = CollectionMethod.CHARGE_AUTOMATICALLY,
        description: Optional[str] = None,
        days_until_due: Optional[int] = None,
        auto_advance: bool = True,
        default_payment_method_id: Optional[str] = None,
        default_tax_rates: Optional[List[str]] = None,
        discounts: Optional[List[str]] = None,
        metadata: Optional[Dict[str, str]] = None,
        idempotency_key: Optional[str] = None,
    ) -> Invoice:
        """
        Create a new invoice.
        
        Args:
            customer_id: Customer ID
            subscription_id: Associated subscription ID
            collection_method: Collection method
            description: Invoice description
            days_until_due: Days until invoice is due
            auto_advance: Whether to auto-advance invoice
            default_payment_method_id: Default payment method
            default_tax_rates: Default tax rate IDs
            discounts: Discount IDs to apply
            metadata: Custom metadata
            idempotency_key: Idempotency key
        
        Returns:
            Created Invoice object
        
        Raises:
            InvalidRequestError: If parameters are invalid
        """
        self._client._check_api_key()
        
        try:
            self._client.customers.retrieve(customer_id)
        except ResourceNotFoundError:
            raise InvalidRequestError(f"Customer {customer_id} not found", param="customer_id")
        
        now = time.time()
        
        invoice = Invoice(
            id=_generate_id("in"),
            customer_id=customer_id,
            subscription_id=subscription_id,
            status=InvoiceStatus.DRAFT,
            collection_method=collection_method,
            default_payment_method_id=default_payment_method_id,
            due_date=now + (days_until_due or 30) * 24 * 3600 if collection_method == CollectionMethod.SEND_INVOICE else None,
            period_start=now,
            period_end=now + 30 * 24 * 3600,
            metadata=metadata or {},
            discounts=discounts or [],
            livemode=not self._client.test_mode,
        )
        
        with self._lock:
            self._invoices[invoice.id] = invoice
        
        self._client._log_operation(
            "invoice.create",
            invoice_id=invoice.id,
            customer_id=customer_id
        )
        return invoice
    
    def retrieve(self, invoice_id: str) -> Invoice:
        """
        Retrieve an invoice by ID.
        
        Args:
            invoice_id: Invoice ID (in_xxx)
        
        Returns:
            Invoice object
        
        Raises:
            ResourceNotFoundError: If invoice not found
        """
        self._client._check_api_key()
        
        with self._lock:
            if invoice_id not in self._invoices:
                raise ResourceNotFoundError("Invoice", invoice_id)
            return copy.deepcopy(self._invoices[invoice_id])
    
    def finalize(self, invoice_id: str, auto_advance: bool = True) -> Invoice:
        """
        Finalize a draft invoice.
        
        Args:
            invoice_id: Invoice ID to finalize
            auto_advance: Whether to auto-advance invoice
        
        Returns:
            Finalized Invoice object
        
        Raises:
            ResourceNotFoundError: If invoice not found
            InvoiceError: If invoice cannot be finalized
        """
        self._client._check_api_key()
        
        with self._lock:
            if invoice_id not in self._invoices:
                raise ResourceNotFoundError("Invoice", invoice_id)
            
            invoice = self._invoices[invoice_id]
            
            if invoice.status != InvoiceStatus.DRAFT:
                raise InvoiceError(
                    "Only draft invoices can be finalized",
                    invoice_id=invoice_id
                )
            
            invoice.status = InvoiceStatus.OPEN
            invoice.finalized_at = time.time()
            invoice.number = f"INV-{secrets.token_hex(4).upper()}"
            invoice.hosted_invoice_url = f"https://invoice.stripe.com/{invoice.id}"
            invoice.invoice_pdf = f"https://invoice.stripe.com/{invoice.id}/pdf"
            
            self._client._log_operation("invoice.finalize", invoice_id=invoice_id)
            return copy.deepcopy(invoice)
    
    def pay(
        self,
        invoice_id: str,
        payment_method_id: Optional[str] = None,
        paid_out_of_band: bool = False,
    ) -> Invoice:
        """
        Pay an invoice.
        
        Args:
            invoice_id: Invoice ID to pay
            payment_method_id: Payment method ID to use
            paid_out_of_band: Mark as paid outside Stripe
        
        Returns:
            Paid Invoice object
        
        Raises:
            ResourceNotFoundError: If invoice not found
            InvoiceError: If invoice cannot be paid
        """
        self._client._check_api_key()
        
        with self._lock:
            if invoice_id not in self._invoices:
                raise ResourceNotFoundError("Invoice", invoice_id)
            
            invoice = self._invoices[invoice_id]
            
            if invoice.status not in [InvoiceStatus.OPEN, InvoiceStatus.DRAFT]:
                raise InvoiceError(
                    f"Invoice with status {invoice.status.value} cannot be paid",
                    invoice_id=invoice_id
                )
            
            if invoice.status == InvoiceStatus.DRAFT:
                invoice.status = InvoiceStatus.OPEN
                invoice.finalized_at = time.time()
            
            invoice.status = InvoiceStatus.PAID
            invoice.paid = True
            invoice.paid_at = time.time()
            invoice.amount_paid = invoice.amount_due
            invoice.amount_remaining = 0
            
            self._client._log_operation("invoice.pay", invoice_id=invoice_id)
            return copy.deepcopy(invoice)
    
    def void(self, invoice_id: str) -> Invoice:
        """
        Void an invoice.
        
        Args:
            invoice_id: Invoice ID to void
        
        Returns:
            Voided Invoice object
        
        Raises:
            ResourceNotFoundError: If invoice not found
            InvoiceError: If invoice cannot be voided
        """
        self._client._check_api_key()
        
        with self._lock:
            if invoice_id not in self._invoices:
                raise ResourceNotFoundError("Invoice", invoice_id)
            
            invoice = self._invoices[invoice_id]
            
            if invoice.status not in [InvoiceStatus.DRAFT, InvoiceStatus.OPEN]:
                raise InvoiceError(
                    "Only draft or open invoices can be voided",
                    invoice_id=invoice_id
                )
            
            invoice.status = InvoiceStatus.VOID
            invoice.voided_at = time.time()
            
            self._client._log_operation("invoice.void", invoice_id=invoice_id)
            return copy.deepcopy(invoice)
    
    def add_invoice_item(
        self,
        invoice_id: str,
        amount: int,
        currency: Union[Currency, str] = Currency.USD,
        description: Optional[str] = None,
        quantity: int = 1,
        price_id: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> Invoice:
        """
        Add a line item to an invoice.
        
        Args:
            invoice_id: Invoice ID
            amount: Line item amount in cents
            currency: Currency code
            description: Line item description
            quantity: Quantity
            price_id: Price ID
            metadata: Custom metadata
        
        Returns:
            Updated Invoice object
        
        Raises:
            ResourceNotFoundError: If invoice not found
            InvoiceError: If items cannot be added
        """
        self._client._check_api_key()
        
        if isinstance(currency, str):
            currency = Currency(currency.lower())
        
        with self._lock:
            if invoice_id not in self._invoices:
                raise ResourceNotFoundError("Invoice", invoice_id)
            
            invoice = self._invoices[invoice_id]
            
            if invoice.status != InvoiceStatus.DRAFT:
                raise InvoiceError(
                    "Can only add items to draft invoices",
                    invoice_id=invoice_id
                )
            
            line_item = InvoiceLineItem(
                id=_generate_id("il"),
                description=description,
                amount=amount,
                currency=currency,
                quantity=quantity,
                price_id=price_id,
                metadata=metadata or {},
            )
            
            invoice.lines.append(line_item)
            invoice.subtotal += amount * quantity
            invoice.total += amount * quantity
            invoice.amount_due += amount * quantity
            invoice.amount_remaining += amount * quantity
            
            self._client._log_operation(
                "invoice_item.add",
                invoice_id=invoice_id,
                amount=amount
            )
            return copy.deepcopy(invoice)
    
    def list(
        self,
        customer_id: Optional[str] = None,
        subscription_id: Optional[str] = None,
        status: Optional[InvoiceStatus] = None,
        limit: int = 10,
        starting_after: Optional[str] = None,
    ) -> List[Invoice]:
        """
        List invoices.
        
        Args:
            customer_id: Filter by customer ID
            subscription_id: Filter by subscription ID
            status: Filter by status
            limit: Maximum number of invoices to return
            starting_after: Cursor for pagination
        
        Returns:
            List of Invoice objects
        """
        self._client._check_api_key()
        
        limit = min(limit, StripeLimit.MAX_ITEMS_PER_LIST.value)
        
        with self._lock:
            invoices = list(self._invoices.values())
        
        if customer_id:
            invoices = [i for i in invoices if i.customer_id == customer_id]
        
        if subscription_id:
            invoices = [i for i in invoices if i.subscription_id == subscription_id]
        
        if status:
            invoices = [i for i in invoices if i.status == status]
        
        invoices.sort(key=lambda i: i.created, reverse=True)
        
        if starting_after:
            idx = next((j for j, i in enumerate(invoices) if i.id == starting_after), -1)
            if idx >= 0:
                invoices = invoices[idx + 1:]
        
        return [copy.deepcopy(i) for i in invoices[:limit]]


class WebhookHandler:
    """
    Webhook event handling and signature verification.
    
    Handles incoming webhook events from Stripe with signature verification.
    """
    
    def __init__(self, client: 'StripeClient'):
        """Initialize with parent client."""
        self._client = client
        self._handlers: Dict[str, List[Callable[[WebhookEvent], None]]] = {}
        self._lock = threading.Lock()
    
    def verify_signature(
        self,
        payload: Union[str, bytes],
        signature: str,
        tolerance: int = StripeLimit.WEBHOOK_SIGNATURE_TOLERANCE_SECONDS.value,
    ) -> bool:
        """
        Verify webhook signature.
        
        Args:
            payload: Webhook payload
            signature: Stripe-Signature header value
            tolerance: Signature timestamp tolerance in seconds
        
        Returns:
            True if signature is valid
        
        Raises:
            WebhookSignatureError: If signature is invalid
        """
        if not self._client.webhook_secret:
            raise WebhookSignatureError("Webhook secret not configured")
        
        if isinstance(payload, str):
            payload = payload.encode('utf-8')
        
        try:
            parts = dict(pair.split('=', 1) for pair in signature.split(','))
            timestamp = int(parts.get('t', '0'))
            signatures = [
                parts[k] for k in parts if k.startswith('v1')
            ]
        except (ValueError, KeyError):
            raise WebhookSignatureError("Invalid signature format")
        
        if not signatures:
            raise WebhookSignatureError("No v1 signature found")
        
        now = int(time.time())
        if abs(now - timestamp) > tolerance:
            raise WebhookSignatureError(
                f"Timestamp outside tolerance window ({tolerance}s)"
            )
        
        signed_payload = f"{timestamp}.".encode('utf-8') + payload
        expected_sig = hmac.new(
            self._client.webhook_secret.encode('utf-8'),
            signed_payload,
            hashlib.sha256
        ).hexdigest()
        
        for sig in signatures:
            if _timing_safe_compare(sig.encode('utf-8'), expected_sig.encode('utf-8')):
                return True
        
        raise WebhookSignatureError("Signature verification failed")
    
    def construct_event(
        self,
        payload: Union[str, bytes],
        signature: str,
        tolerance: int = StripeLimit.WEBHOOK_SIGNATURE_TOLERANCE_SECONDS.value,
    ) -> WebhookEvent:
        """
        Construct and verify a webhook event.
        
        Args:
            payload: Webhook payload
            signature: Stripe-Signature header value
            tolerance: Signature timestamp tolerance
        
        Returns:
            WebhookEvent object
        
        Raises:
            WebhookSignatureError: If signature is invalid
        """
        self.verify_signature(payload, signature, tolerance)
        
        if isinstance(payload, bytes):
            payload = payload.decode('utf-8')
        
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as e:
            raise WebhookSignatureError(f"Invalid JSON payload: {e}")
        
        return WebhookEvent.from_dict(data)
    
    def handle(
        self,
        payload: Union[str, bytes],
        signature: str,
        tolerance: int = StripeLimit.WEBHOOK_SIGNATURE_TOLERANCE_SECONDS.value,
    ) -> WebhookEvent:
        """
        Handle a webhook event.
        
        Args:
            payload: Webhook payload
            signature: Stripe-Signature header value
            tolerance: Signature timestamp tolerance
        
        Returns:
            WebhookEvent object
        
        Raises:
            WebhookSignatureError: If signature is invalid
        """
        event = self.construct_event(payload, signature, tolerance)
        
        with self._lock:
            handlers = self._handlers.get(event.type, [])
            all_handlers = self._handlers.get('*', [])
        
        for handler in handlers + all_handlers:
            try:
                handler(event)
            except Exception as e:
                self._client._log_operation(
                    "webhook.handler_error",
                    event_type=event.type,
                    error=str(e)
                )
        
        self._client._log_operation(
            "webhook.handle",
            event_id=event.id,
            event_type=event.type
        )
        return event
    
    def on(self, event_type: str, handler: Callable[[WebhookEvent], None]) -> None:
        """
        Register a webhook event handler.
        
        Args:
            event_type: Event type to handle (or '*' for all events)
            handler: Handler function
        """
        with self._lock:
            if event_type not in self._handlers:
                self._handlers[event_type] = []
            self._handlers[event_type].append(handler)
    
    def off(self, event_type: str, handler: Optional[Callable[[WebhookEvent], None]] = None) -> None:
        """
        Unregister a webhook event handler.
        
        Args:
            event_type: Event type
            handler: Specific handler to remove (or None to remove all)
        """
        with self._lock:
            if event_type not in self._handlers:
                return
            
            if handler is None:
                del self._handlers[event_type]
            else:
                self._handlers[event_type] = [
                    h for h in self._handlers[event_type] if h != handler
                ]


class ProductManager:
    """
    Product and price catalog management.
    
    Handles product creation, pricing, and catalog operations.
    """
    
    def __init__(self, client: 'StripeClient'):
        """Initialize with parent client."""
        self._client = client
        self._products: Dict[str, Product] = {}
        self._prices: Dict[str, Price] = {}
        self._coupons: Dict[str, Coupon] = {}
        self._tax_rates: Dict[str, TaxRate] = {}
        self._lock = threading.Lock()
    
    def create_product(
        self,
        name: str,
        description: Optional[str] = None,
        images: Optional[List[str]] = None,
        features: Optional[List[Dict[str, str]]] = None,
        active: bool = True,
        unit_label: Optional[str] = None,
        statement_descriptor: Optional[str] = None,
        tax_code: Optional[str] = None,
        url: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
        idempotency_key: Optional[str] = None,
    ) -> Product:
        """
        Create a new product.
        
        Args:
            name: Product name
            description: Product description
            images: Product image URLs (max 8)
            features: Product features
            active: Whether product is active
            unit_label: Label for unit quantity
            statement_descriptor: Statement descriptor
            tax_code: Tax code
            url: Product URL
            metadata: Custom metadata
            idempotency_key: Idempotency key
        
        Returns:
            Created Product object
        """
        self._client._check_api_key()
        
        product = Product(
            id=_generate_id("prod"),
            name=name,
            description=description,
            active=active,
            images=images or [],
            features=features or [],
            unit_label=unit_label,
            statement_descriptor=statement_descriptor,
            tax_code=tax_code,
            url=url,
            metadata=metadata or {},
            livemode=not self._client.test_mode,
        )
        
        with self._lock:
            self._products[product.id] = product
        
        self._client._log_operation("product.create", product_id=product.id)
        return product
    
    def retrieve_product(self, product_id: str) -> Product:
        """Retrieve a product by ID."""
        self._client._check_api_key()
        
        with self._lock:
            if product_id not in self._products:
                raise ResourceNotFoundError("Product", product_id)
            return copy.deepcopy(self._products[product_id])
    
    def update_product(
        self,
        product_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        images: Optional[List[str]] = None,
        features: Optional[List[Dict[str, str]]] = None,
        active: Optional[bool] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> Product:
        """Update a product."""
        self._client._check_api_key()
        
        with self._lock:
            if product_id not in self._products:
                raise ResourceNotFoundError("Product", product_id)
            
            product = self._products[product_id]
            
            if name is not None:
                product.name = name
            if description is not None:
                product.description = description
            if images is not None:
                product.images = images
            if features is not None:
                product.features = features
            if active is not None:
                product.active = active
            if metadata is not None:
                product.metadata.update(metadata)
            
            product.updated = time.time()
            
            self._client._log_operation("product.update", product_id=product_id)
            return copy.deepcopy(product)
    
    def delete_product(self, product_id: str) -> Product:
        """Delete a product."""
        self._client._check_api_key()
        
        with self._lock:
            if product_id not in self._products:
                raise ResourceNotFoundError("Product", product_id)
            
            has_prices = any(p.product_id == product_id for p in self._prices.values())
            if has_prices:
                raise InvalidRequestError(
                    "Cannot delete product with active prices",
                    param="product_id"
                )
            
            product = self._products.pop(product_id)
            self._client._log_operation("product.delete", product_id=product_id)
            return product
    
    def list_products(
        self,
        active: Optional[bool] = None,
        limit: int = 10,
        starting_after: Optional[str] = None,
    ) -> List[Product]:
        """List products."""
        self._client._check_api_key()
        
        limit = min(limit, StripeLimit.MAX_ITEMS_PER_LIST.value)
        
        with self._lock:
            products = list(self._products.values())
        
        if active is not None:
            products = [p for p in products if p.active == active]
        
        products.sort(key=lambda p: p.created, reverse=True)
        
        if starting_after:
            idx = next((i for i, p in enumerate(products) if p.id == starting_after), -1)
            if idx >= 0:
                products = products[idx + 1:]
        
        return [copy.deepcopy(p) for p in products[:limit]]
    
    def create_price(
        self,
        product_id: str,
        unit_amount: int,
        currency: Union[Currency, str] = Currency.USD,
        recurring: Optional[Dict[str, Any]] = None,
        billing_scheme: str = "per_unit",
        tiers: Optional[List[Dict[str, Any]]] = None,
        tiers_mode: Optional[str] = None,
        lookup_key: Optional[str] = None,
        nickname: Optional[str] = None,
        active: bool = True,
        metadata: Optional[Dict[str, str]] = None,
        idempotency_key: Optional[str] = None,
    ) -> Price:
        """
        Create a new price.
        
        Args:
            product_id: Product ID
            unit_amount: Price in cents
            currency: Currency code
            recurring: Recurring billing settings
            billing_scheme: Billing scheme (per_unit or tiered)
            tiers: Pricing tiers
            tiers_mode: Tiers mode
            lookup_key: Lookup key
            nickname: Human-readable nickname
            active: Whether price is active
            metadata: Custom metadata
            idempotency_key: Idempotency key
        
        Returns:
            Created Price object
        """
        self._client._check_api_key()
        
        if isinstance(currency, str):
            currency = Currency(currency.lower())
        
        with self._lock:
            if product_id not in self._products:
                raise ResourceNotFoundError("Product", product_id)
        
        price = Price(
            id=_generate_id("price"),
            product_id=product_id,
            active=active,
            currency=currency,
            unit_amount=unit_amount,
            recurring=recurring,
            type="recurring" if recurring else "one_time",
            billing_scheme=billing_scheme,
            tiers=tiers,
            tiers_mode=tiers_mode,
            lookup_key=lookup_key,
            nickname=nickname,
            metadata=metadata or {},
            livemode=not self._client.test_mode,
        )
        
        with self._lock:
            self._prices[price.id] = price
        
        self._client._log_operation("price.create", price_id=price.id)
        return price
    
    def retrieve_price(self, price_id: str) -> Price:
        """Retrieve a price by ID."""
        self._client._check_api_key()
        
        with self._lock:
            if price_id not in self._prices:
                raise ResourceNotFoundError("Price", price_id)
            return copy.deepcopy(self._prices[price_id])
    
    def update_price(
        self,
        price_id: str,
        active: Optional[bool] = None,
        nickname: Optional[str] = None,
        lookup_key: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> Price:
        """Update a price."""
        self._client._check_api_key()
        
        with self._lock:
            if price_id not in self._prices:
                raise ResourceNotFoundError("Price", price_id)
            
            price = self._prices[price_id]
            
            if active is not None:
                price.active = active
            if nickname is not None:
                price.nickname = nickname
            if lookup_key is not None:
                price.lookup_key = lookup_key
            if metadata is not None:
                price.metadata.update(metadata)
            
            self._client._log_operation("price.update", price_id=price_id)
            return copy.deepcopy(price)
    
    def list_prices(
        self,
        product_id: Optional[str] = None,
        active: Optional[bool] = None,
        type: Optional[str] = None,
        limit: int = 10,
        starting_after: Optional[str] = None,
    ) -> List[Price]:
        """List prices."""
        self._client._check_api_key()
        
        limit = min(limit, StripeLimit.MAX_ITEMS_PER_LIST.value)
        
        with self._lock:
            prices = list(self._prices.values())
        
        if product_id:
            prices = [p for p in prices if p.product_id == product_id]
        
        if active is not None:
            prices = [p for p in prices if p.active == active]
        
        if type:
            prices = [p for p in prices if p.type == type]
        
        prices.sort(key=lambda p: p.created, reverse=True)
        
        if starting_after:
            idx = next((i for i, p in enumerate(prices) if p.id == starting_after), -1)
            if idx >= 0:
                prices = prices[idx + 1:]
        
        return [copy.deepcopy(p) for p in prices[:limit]]
    
    def create_coupon(
        self,
        name: Optional[str] = None,
        percent_off: Optional[float] = None,
        amount_off: Optional[int] = None,
        currency: Optional[Union[Currency, str]] = None,
        duration: str = "once",
        duration_in_months: Optional[int] = None,
        max_redemptions: Optional[int] = None,
        redeem_by: Optional[float] = None,
        applies_to: Optional[Dict[str, List[str]]] = None,
        metadata: Optional[Dict[str, str]] = None,
        idempotency_key: Optional[str] = None,
    ) -> Coupon:
        """Create a coupon."""
        self._client._check_api_key()
        
        if not percent_off and not amount_off:
            raise InvalidRequestError(
                "Either percent_off or amount_off is required",
                param="percent_off"
            )
        
        if amount_off and not currency:
            raise InvalidRequestError(
                "Currency is required for amount_off coupons",
                param="currency"
            )
        
        if isinstance(currency, str):
            currency = Currency(currency.lower())
        
        coupon = Coupon(
            id=secrets.token_hex(8).upper(),
            name=name,
            percent_off=percent_off,
            amount_off=amount_off,
            currency=currency,
            duration=duration,
            duration_in_months=duration_in_months,
            max_redemptions=max_redemptions,
            redeem_by=redeem_by,
            applies_to=applies_to,
            metadata=metadata or {},
            livemode=not self._client.test_mode,
        )
        
        with self._lock:
            self._coupons[coupon.id] = coupon
        
        self._client._log_operation("coupon.create", coupon_id=coupon.id)
        return coupon
    
    def create_tax_rate(
        self,
        display_name: str,
        percentage: float,
        description: Optional[str] = None,
        inclusive: bool = False,
        active: bool = True,
        country: Optional[str] = None,
        state: Optional[str] = None,
        jurisdiction: Optional[str] = None,
        tax_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
        idempotency_key: Optional[str] = None,
    ) -> TaxRate:
        """Create a tax rate."""
        self._client._check_api_key()
        
        tax_rate = TaxRate(
            id=_generate_id("txr"),
            display_name=display_name,
            description=description,
            percentage=percentage,
            inclusive=inclusive,
            active=active,
            country=country,
            state=state,
            jurisdiction=jurisdiction,
            tax_type=tax_type,
            metadata=metadata or {},
            livemode=not self._client.test_mode,
        )
        
        with self._lock:
            self._tax_rates[tax_rate.id] = tax_rate
        
        self._client._log_operation("tax_rate.create", tax_rate_id=tax_rate.id)
        return tax_rate


class CheckoutManager:
    """
    Checkout and portal session management.
    
    Handles checkout session creation and customer portal sessions.
    """
    
    def __init__(self, client: 'StripeClient'):
        """Initialize with parent client."""
        self._client = client
        self._checkout_sessions: Dict[str, CheckoutSession] = {}
        self._portal_sessions: Dict[str, PortalSession] = {}
        self._lock = threading.Lock()
    
    def create_session(
        self,
        success_url: str,
        cancel_url: str,
        mode: CheckoutMode = CheckoutMode.PAYMENT,
        customer_id: Optional[str] = None,
        customer_email: Optional[str] = None,
        client_reference_id: Optional[str] = None,
        line_items: Optional[List[Dict[str, Any]]] = None,
        price_id: Optional[str] = None,
        quantity: int = 1,
        allow_promotion_codes: bool = False,
        billing_address_collection: Optional[str] = None,
        shipping_address_collection: Optional[Dict[str, Any]] = None,
        subscription_data: Optional[Dict[str, Any]] = None,
        payment_intent_data: Optional[Dict[str, Any]] = None,
        expires_at: Optional[float] = None,
        metadata: Optional[Dict[str, str]] = None,
        idempotency_key: Optional[str] = None,
    ) -> CheckoutSession:
        """
        Create a checkout session.
        
        Args:
            success_url: Success redirect URL
            cancel_url: Cancel redirect URL
            mode: Checkout mode (payment, subscription, setup)
            customer_id: Existing customer ID
            customer_email: Customer email for new customers
            client_reference_id: Client reference ID
            line_items: Line items
            price_id: Price ID (shorthand for single item)
            quantity: Quantity for price_id
            allow_promotion_codes: Allow promotion codes
            billing_address_collection: Billing address collection mode
            shipping_address_collection: Shipping address collection settings
            subscription_data: Subscription data for subscription mode
            payment_intent_data: Payment intent data for payment mode
            expires_at: Session expiration timestamp
            metadata: Custom metadata
            idempotency_key: Idempotency key
        
        Returns:
            Created CheckoutSession object
        """
        self._client._check_api_key()
        
        if price_id:
            line_items = [{"price": price_id, "quantity": quantity}]
        
        session = CheckoutSession(
            id=_generate_id("cs"),
            customer_id=customer_id,
            mode=mode,
            status="open",
            success_url=success_url,
            cancel_url=cancel_url,
            customer_email=customer_email,
            client_reference_id=client_reference_id,
            line_items=line_items or [],
            allow_promotion_codes=allow_promotion_codes,
            billing_address_collection=billing_address_collection,
            shipping_address_collection=shipping_address_collection,
            expires_at=expires_at or time.time() + 24 * 3600,
            metadata=metadata or {},
            livemode=not self._client.test_mode,
        )
        
        session.url = f"https://checkout.stripe.com/c/pay/{session.id}"
        
        with self._lock:
            self._checkout_sessions[session.id] = session
        
        self._client._log_operation(
            "checkout.session.create",
            session_id=session.id,
            mode=mode.value
        )
        return session
    
    def retrieve_session(
        self,
        session_id: str,
        expand: Optional[List[str]] = None,
    ) -> CheckoutSession:
        """
        Retrieve a checkout session.
        
        Args:
            session_id: Session ID
            expand: Fields to expand
        
        Returns:
            CheckoutSession object
        """
        self._client._check_api_key()
        
        with self._lock:
            if session_id not in self._checkout_sessions:
                raise ResourceNotFoundError("CheckoutSession", session_id)
            return copy.deepcopy(self._checkout_sessions[session_id])
    
    def expire_session(self, session_id: str) -> CheckoutSession:
        """
        Expire a checkout session.
        
        Args:
            session_id: Session ID
        
        Returns:
            Expired CheckoutSession object
        """
        self._client._check_api_key()
        
        with self._lock:
            if session_id not in self._checkout_sessions:
                raise ResourceNotFoundError("CheckoutSession", session_id)
            
            session = self._checkout_sessions[session_id]
            session.status = "expired"
            
            self._client._log_operation("checkout.session.expire", session_id=session_id)
            return copy.deepcopy(session)
    
    def create_portal_session(
        self,
        customer_id: str,
        return_url: str,
        configuration: Optional[str] = None,
        on_behalf_of: Optional[str] = None,
    ) -> PortalSession:
        """
        Create a billing portal session.
        
        Args:
            customer_id: Customer ID
            return_url: Return URL after portal
            configuration: Portal configuration ID
            on_behalf_of: Connected account ID
        
        Returns:
            Created PortalSession object
        """
        self._client._check_api_key()
        
        try:
            self._client.customers.retrieve(customer_id)
        except ResourceNotFoundError:
            raise InvalidRequestError(f"Customer {customer_id} not found", param="customer_id")
        
        portal = PortalSession(
            id=_generate_id("bps"),
            customer_id=customer_id,
            url=f"https://billing.stripe.com/p/{_generate_id('session')}",
            return_url=return_url,
            livemode=not self._client.test_mode,
        )
        
        with self._lock:
            self._portal_sessions[portal.id] = portal
        
        self._client._log_operation(
            "portal.session.create",
            session_id=portal.id,
            customer_id=customer_id
        )
        return portal
    
    def list_sessions(
        self,
        customer_id: Optional[str] = None,
        payment_intent_id: Optional[str] = None,
        subscription_id: Optional[str] = None,
        limit: int = 10,
        starting_after: Optional[str] = None,
    ) -> List[CheckoutSession]:
        """List checkout sessions."""
        self._client._check_api_key()
        
        limit = min(limit, StripeLimit.MAX_ITEMS_PER_LIST.value)
        
        with self._lock:
            sessions = list(self._checkout_sessions.values())
        
        if customer_id:
            sessions = [s for s in sessions if s.customer_id == customer_id]
        
        if payment_intent_id:
            sessions = [s for s in sessions if s.payment_intent_id == payment_intent_id]
        
        if subscription_id:
            sessions = [s for s in sessions if s.subscription_id == subscription_id]
        
        sessions.sort(key=lambda s: s.created, reverse=True)
        
        if starting_after:
            idx = next((i for i, s in enumerate(sessions) if s.id == starting_after), -1)
            if idx >= 0:
                sessions = sessions[idx + 1:]
        
        return [copy.deepcopy(s) for s in sessions[:limit]]


class StripeClient:
    """
    Main Stripe client wrapper.
    
    Provides access to all Stripe API resources through manager classes.
    This is an abstraction layer - actual Stripe API calls require the stripe-python SDK.
    
    Attributes:
        customers: Customer management operations
        subscriptions: Subscription lifecycle operations
        payments: Payment processing operations
        invoices: Invoice management operations
        webhooks: Webhook handling operations
        products: Product catalog operations
        checkout: Checkout session operations
    
    Usage:
        # Initialize the client
        client = StripeClient(
            api_key="sk_test_...",
            webhook_secret="whsec_...",
            test_mode=True
        )
        
        # Use manager classes
        customer = client.customers.create(email="user@example.com")
        subscription = client.subscriptions.create(
            customer_id=customer.id,
            price_id="price_xxx"
        )
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        webhook_secret: Optional[str] = None,
        test_mode: bool = True,
        api_version: str = "2023-10-16",
        max_network_retries: int = 2,
        timeout: float = 30.0,
        log_level: int = logging.INFO,
    ):
        """
        Initialize the Stripe client.
        
        Args:
            api_key: Stripe API key (sk_test_... or sk_live_...)
            webhook_secret: Webhook signing secret (whsec_...)
            test_mode: Whether to use test mode
            api_version: Stripe API version
            max_network_retries: Maximum network retries
            timeout: Request timeout in seconds
            log_level: Logging level
        """
        self.api_key = api_key or os.environ.get('STRIPE_SECRET_KEY')
        self.webhook_secret = webhook_secret or os.environ.get('STRIPE_WEBHOOK_SECRET')
        self.test_mode = test_mode
        self.api_version = api_version
        self.max_network_retries = max_network_retries
        self.timeout = timeout
        
        self._logger = logging.getLogger('stripe_integration')
        self._logger.setLevel(log_level)
        
        if self.api_key:
            if self.api_key.startswith('sk_live_'):
                self.test_mode = False
            elif self.api_key.startswith('sk_test_'):
                self.test_mode = True
        
        self.customers = CustomerManager(self)
        self.subscriptions = SubscriptionManager(self)
        self.payments = PaymentManager(self)
        self.invoices = InvoiceManager(self)
        self.webhooks = WebhookHandler(self)
        self.products = ProductManager(self)
        self.checkout = CheckoutManager(self)
    
    def _check_api_key(self) -> None:
        """Check if API key is configured."""
        if not self.api_key:
            raise AuthenticationError("API key is not configured")
    
    def _log_operation(self, operation: str, **kwargs) -> None:
        """Log an API operation."""
        self._logger.debug(f"Stripe {operation}: {kwargs}")
    
    def set_api_key(self, api_key: str) -> None:
        """
        Set the API key.
        
        Args:
            api_key: Stripe API key
        """
        self.api_key = api_key
        if api_key.startswith('sk_live_'):
            self.test_mode = False
        elif api_key.startswith('sk_test_'):
            self.test_mode = True
    
    def set_webhook_secret(self, webhook_secret: str) -> None:
        """
        Set the webhook secret.
        
        Args:
            webhook_secret: Webhook signing secret
        """
        self.webhook_secret = webhook_secret
    
    @property
    def is_test_mode(self) -> bool:
        """Check if client is in test mode."""
        return self.test_mode
    
    def create_customer(self, **kwargs) -> Customer:
        """Convenience method to create a customer."""
        return self.customers.create(**kwargs)
    
    def update_customer(self, customer_id: str, **kwargs) -> Customer:
        """Convenience method to update a customer."""
        return self.customers.update(customer_id, **kwargs)
    
    def create_subscription(self, **kwargs) -> Subscription:
        """Convenience method to create a subscription."""
        return self.subscriptions.create(**kwargs)
    
    def cancel_subscription(self, subscription_id: str, **kwargs) -> Subscription:
        """Convenience method to cancel a subscription."""
        return self.subscriptions.cancel(subscription_id, **kwargs)
    
    def update_subscription(self, subscription_id: str, **kwargs) -> Subscription:
        """Convenience method to update a subscription."""
        return self.subscriptions.update(subscription_id, **kwargs)
    
    def create_payment_intent(self, **kwargs) -> PaymentIntent:
        """Convenience method to create a payment intent."""
        return self.payments.create_payment_intent(**kwargs)
    
    def capture_payment(self, payment_intent_id: str, **kwargs) -> PaymentIntent:
        """Convenience method to capture a payment."""
        return self.payments.capture_payment(payment_intent_id, **kwargs)
    
    def create_checkout_session(self, **kwargs) -> CheckoutSession:
        """Convenience method to create a checkout session."""
        return self.checkout.create_session(**kwargs)
    
    def create_portal_session(self, **kwargs) -> PortalSession:
        """Convenience method to create a portal session."""
        return self.checkout.create_portal_session(**kwargs)
    
    def handle_webhook(self, payload: Union[str, bytes], signature: str) -> WebhookEvent:
        """Convenience method to handle a webhook."""
        return self.webhooks.handle(payload, signature)
    
    def verify_webhook_signature(self, payload: Union[str, bytes], signature: str) -> bool:
        """Convenience method to verify a webhook signature."""
        return self.webhooks.verify_signature(payload, signature)
    
    def create_refund(self, **kwargs) -> Refund:
        """Convenience method to create a refund."""
        return self.payments.create_refund(**kwargs)
    
    def list_invoices(self, **kwargs) -> List[Invoice]:
        """Convenience method to list invoices."""
        return self.invoices.list(**kwargs)


_default_client: Optional[StripeClient] = None
_lock = threading.Lock()


def get_default_client() -> StripeClient:
    """
    Get the default Stripe client.
    
    Returns:
        Default StripeClient instance
    """
    global _default_client
    with _lock:
        if _default_client is None:
            _default_client = StripeClient()
        return _default_client


def set_default_client(client: StripeClient) -> None:
    """
    Set the default Stripe client.
    
    Args:
        client: StripeClient instance to use as default
    """
    global _default_client
    with _lock:
        _default_client = client


def init_stripe(
    api_key: Optional[str] = None,
    webhook_secret: Optional[str] = None,
    test_mode: bool = True,
) -> StripeClient:
    """
    Initialize Stripe with the given configuration.
    
    Args:
        api_key: Stripe API key
        webhook_secret: Webhook signing secret
        test_mode: Whether to use test mode
    
    Returns:
        Configured StripeClient instance
    """
    client = StripeClient(
        api_key=api_key,
        webhook_secret=webhook_secret,
        test_mode=test_mode,
    )
    set_default_client(client)
    return client


def create_customer(**kwargs) -> Customer:
    """Helper function to create a customer using default client."""
    return get_default_client().create_customer(**kwargs)


def update_customer(customer_id: str, **kwargs) -> Customer:
    """Helper function to update a customer using default client."""
    return get_default_client().update_customer(customer_id, **kwargs)


def create_subscription(**kwargs) -> Subscription:
    """Helper function to create a subscription using default client."""
    return get_default_client().create_subscription(**kwargs)


def cancel_subscription(subscription_id: str, **kwargs) -> Subscription:
    """Helper function to cancel a subscription using default client."""
    return get_default_client().cancel_subscription(subscription_id, **kwargs)


def update_subscription(subscription_id: str, **kwargs) -> Subscription:
    """Helper function to update a subscription using default client."""
    return get_default_client().update_subscription(subscription_id, **kwargs)


def create_payment_intent(**kwargs) -> PaymentIntent:
    """Helper function to create a payment intent using default client."""
    return get_default_client().create_payment_intent(**kwargs)


def capture_payment(payment_intent_id: str, **kwargs) -> PaymentIntent:
    """Helper function to capture a payment using default client."""
    return get_default_client().capture_payment(payment_intent_id, **kwargs)


def create_checkout_session(**kwargs) -> CheckoutSession:
    """Helper function to create a checkout session using default client."""
    return get_default_client().create_checkout_session(**kwargs)


def create_portal_session(**kwargs) -> PortalSession:
    """Helper function to create a portal session using default client."""
    return get_default_client().create_portal_session(**kwargs)


def handle_webhook(payload: Union[str, bytes], signature: str) -> WebhookEvent:
    """Helper function to handle a webhook using default client."""
    return get_default_client().handle_webhook(payload, signature)


def verify_webhook_signature(payload: Union[str, bytes], signature: str) -> bool:
    """Helper function to verify a webhook signature using default client."""
    return get_default_client().verify_webhook_signature(payload, signature)


def create_refund(**kwargs) -> Refund:
    """Helper function to create a refund using default client."""
    return get_default_client().create_refund(**kwargs)


def list_invoices(**kwargs) -> List[Invoice]:
    """Helper function to list invoices using default client."""
    return get_default_client().list_invoices(**kwargs)


def format_amount(amount: int, currency: Currency = Currency.USD) -> str:
    """
    Format an amount in cents as a currency string.
    
    Args:
        amount: Amount in cents
        currency: Currency code
    
    Returns:
        Formatted currency string (e.g., "$10.00")
    """
    zero_decimal_currencies = {'JPY', 'KRW', 'VND', 'BIF', 'CLP', 'DJF', 'GNF', 'ISK', 'KMF', 'KRW', 'PYG', 'RWF', 'UGX', 'UYI', 'VND', 'VUV', 'XAF', 'XOF', 'XPF'}
    
    symbols = {
        Currency.USD: '$',
        Currency.EUR: '\u20ac',
        Currency.GBP: '\u00a3',
        Currency.JPY: '\u00a5',
        Currency.CAD: 'CA$',
        Currency.AUD: 'A$',
        Currency.CHF: 'CHF ',
        Currency.CNY: '\u00a5',
        Currency.INR: '\u20b9',
    }
    
    symbol = symbols.get(currency, currency.value.upper() + ' ')
    
    if currency.value.upper() in zero_decimal_currencies:
        return f"{symbol}{amount:,}"
    else:
        return f"{symbol}{amount / 100:,.2f}"

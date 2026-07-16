from lexi_service.security.auth import principal_from_asgi_scope, principal_from_grpc_auth_context


def test_http_headers_cannot_create_a_service_identity():
    assert principal_from_asgi_scope({"headers": [(b"x-tenant", b"other")]}) is None


def test_grpc_requires_tls_and_certificate_common_name():
    assert principal_from_grpc_auth_context({"x509_common_name": [b"svc"]}) is None
    assert principal_from_grpc_auth_context({"transport_security_type": [b"ssl"]}) is None

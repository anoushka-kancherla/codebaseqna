def login(username, password):
    if check_credentials(username, password):
        return create_session(username)
    return None


def check_credentials(username, password):
    return username == "admin" and password == "secret"


def create_session(username):
    return {"user": username, "token": "abc123"}

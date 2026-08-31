"""Create the first administrator account, interactively.

There is no default administrator and no seeded password anywhere in this repository. A
shipped credential is a credential every deployment shares and most never change, so the
first account is made deliberately, by a person, on the machine that will run the
application:

    docker compose exec api python create_admin.py

The password is read with `getpass`, so it is not echoed to the terminal and does not reach
the shell history. It is hashed with Argon2id before it touches the session and is never
printed, never logged and never written anywhere but the `password_hash` column. Taking it
from an argument or an environment variable instead would leave it in `ps` output, in a
history file, or in a compose file someone commits.

Run against an existing address, this refuses rather than overwriting. Promoting or
resetting an existing account is a different operation with different consequences, and a
bootstrap script that silently did either would be a password reset anybody with shell
access could perform by accident.
"""

import getpass
import sys

from sqlalchemy import select

from app.db.models import USER_ROLE_ADMIN, User
from app.db.session import SessionLocal
from app.web_auth import hash_password, normalize_email

# Short, and not a policy. A real password policy belongs with the account management this
# task explicitly does not build; what this floor prevents is the empty or one-character
# password a distracted operator would otherwise create the first administrator with.
MINIMUM_PASSWORD_LENGTH = 12


def prompt_email() -> str:
    """Read the address, normalized the same way login will normalize it."""
    email = normalize_email(input("Email: "))

    if not email:
        sys.exit("An email address is required.")

    return email


def prompt_password() -> str:
    """Read the password twice, without echoing it, and confirm the two agree.

    The confirmation is not ceremony. This password is not recoverable — nothing stores it —
    so a typo in the only administrator's password means creating another account to get back
    in, and the second prompt is what catches it while that is still cheap.
    """
    password = getpass.getpass("Password: ")

    if len(password) < MINIMUM_PASSWORD_LENGTH:
        sys.exit(f"The password must be at least {MINIMUM_PASSWORD_LENGTH} characters.")

    if password != getpass.getpass("Confirm password: "):
        sys.exit("The passwords do not match.")

    return password


def main() -> None:
    """Create one administrator, or explain why it did not."""
    email = prompt_email()

    with SessionLocal() as session:
        existing = session.execute(
            select(User.id).where(User.email == email)
        ).scalar_one_or_none()

        if existing is not None:
            sys.exit(f"An account already exists for {email}.")

        password = prompt_password()

        user = User(
            email=email,
            password_hash=hash_password(password),
            role=USER_ROLE_ADMIN,
        )
        session.add(user)
        session.commit()

        # The address and the role, which is what an operator needs to confirm the right
        # account was made. Never the password, and never its hash.
        print(f"Created administrator {user.email} ({user.id}).")


if __name__ == "__main__":
    main()

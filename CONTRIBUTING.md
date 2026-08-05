# Contributing to blackbox-optimizer

Thanks for considering a contribution — issues and PRs are welcome. This is a single-maintainer
open-source project, so response times may vary.

## Getting started

```
composer install
```

Requires PHP 8.3+ (CI also runs against 8.4). Framework-agnostic — no external services needed to
develop or test.

## Before opening a PR

These are the checks CI runs; running them locally first saves a review round-trip:

```
composer validate --no-check-publish
vendor/bin/phpunit
vendor/bin/phpcs
vendor/bin/phpstan analyse -c phpstan.neon src/
vendor/bin/phpmd src text phpmd.xml
```

`composer rector-dry-run` is also available locally to check for suggested modernizations (not
currently a required CI check).

## Making a change

- Keep PRs focused — one change per PR.
- Branch from and target `main`; branches are merged via squash, so intermediate commit messages
  don't need to be polished.
- Add or adjust PHPUnit tests for any behavior change — this package has no integration-environment
  dependency, so tests are the primary correctness signal.
- Update `README.md` if the public interface or behavior changes.

## Reporting bugs or requesting features

Use the issue templates — they ask for the information needed to reproduce a bug or evaluate a
request. For security issues, see [SECURITY.md](SECURITY.md) instead of opening a public issue.

## License

By contributing, you agree your contribution is licensed under this project's [MIT license](LICENSE).

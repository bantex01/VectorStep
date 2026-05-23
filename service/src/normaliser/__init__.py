from .alertmanager import AlertmanagerParser
from .generic import GenericParser

PARSERS: dict = {
    "alertmanager": AlertmanagerParser,
    "generic": GenericParser,
}

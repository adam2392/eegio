import logging
from abc import ABC

logger = logging.getLogger(__name__)


class BaseWrite(ABC):
    def _check_info(self, info):
        for item in info["chs"]:
            ch_name = item["ch_name"]

            if ch_name not in info["ch_names"]:
                logger.log(ch_name)
                logger.log(info["ch_names"])
                # continue
                raise ValueError("{} not in ch names?".format(ch_name))

        for item in info["bads"]:
            if item not in info["ch_names"]:
                logger.log(item)
                # continue
                raise ValueError("{} not in ch names?".format(item))

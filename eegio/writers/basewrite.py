from abc import ABC


class BaseWrite(ABC):
    def _check_info(self, info):
        for item in info["chs"]:
            ch_name = item["ch_name"]

            if ch_name not in info["ch_names"]:
                print(ch_name)
                print(info["ch_names"])
                # continue
                raise ValueError("{} not in ch names?".format(ch_name))

        for item in info["bads"]:
            if item not in info["ch_names"]:
                print(item)
                # continue
                raise ValueError("{} not in ch names?".format(item))

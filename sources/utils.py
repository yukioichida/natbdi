import logging
import sys
import transformers


def setup_logger():
    logging.getLogger("scienceworld").setLevel(logging.CRITICAL)
    logging.getLogger("py4j").setLevel(logging.CRITICAL)

    transformers.logging.set_verbosity_warning()

    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    handlers = [stdout_handler]
    logging.basicConfig(
        level=logging.INFO,
        # format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
        format='[%(asctime)s] %(levelname)s - %(message)s',
        handlers=handlers
    )

    return logging.getLogger(__name__)

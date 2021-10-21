import coloredlogs, logging

datagen_logger = logging.getLogger("DATA_GEN")

coloredlogs.install(fmt='%(asctime)s %(name)s %(levelname)s %(message)s')

def DATA_GEN_LOG_DEBUG(msg):
    datagen_logger.debug(msg)

def DATA_GEN_LOG_INFO(msg):
    datagen_logger.info(msg)

def DATA_GEN_LOG_WARNING(msg):
    datagen_logger.warning(msg)

def DATA_GEN_LOG_ERROR(msg):
    datagen_logger.error(msg)

def DATA_GEN_LOG_CRITICAL(msg):
    datagen_logger.critical(msg)
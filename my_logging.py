import logging
# Gets or creates a logger
logger = logging.getLogger(__name__)  

# set log level
logger.setLevel(logging.DEBUG)

# define file handler and set formatter
file_handler = logging.FileHandler("main.log")
formatter    = logging.Formatter('%(asctime)s : %(levelname)s : %(filename)s:  %(funcName)s : %(lineno)d : %(message)s')
file_handler.setFormatter(formatter)

# add file handler to logger
logger.addHandler(file_handler)
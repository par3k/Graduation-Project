import os
from App import create_app, db
from flask_script import Manager, Shell
from flask_migrate import Migrate, MigrateCommand
from App.model import *

app = create_app()
manager = Manager(app)
migrate = Migrate(app, db)

# def make_shell_context():
manager.add_command('db', MigrateCommand)

# @manager.command
# def test():
#     """Run the unit tests"""
#     import unittest
#     test = unittest.TestLoader().discover('tests')
#     unittest.TextTestRunner(verbosity=2).run(tests)

if __name__ == '__main__':
    #交互环境用
    manager.run()


    #app.run()
#运行程序用

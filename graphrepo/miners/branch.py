# Copyright 2020 NullConvergence
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This module mines branches and contains all related Neo4j queries"""

from graphrepo.miners.default import DefaultMiner
from graphrepo.miners.utils import format_commit_id_date as fcid


class BranchMiner(DefaultMiner):
    """This class holds queries for branches"""

    def query(self, **kwargs):
        """Queries branches by any arguments given in kwargs
        For example kwargs can be {'hash': 'example-hash'}
        :param kwargs: any parameter and value, between hash, name or email
        :returns: list of branch nodes matched
        """
        com_ = self.node_matcher.match("Branch", **kwargs)
        return [dict(x) for x in com_]    

    def get_all(self,):
        """Returns all branches
        :returns: list of branch nodes
        """
        com_ = self.node_matcher.match("Branch")
        return [dict(x) for x in com_]

    def get_commits(self, branch_hash):
        """Returns the commits belonging to a branch
        :param branch_hash: optional; if given, it will
        return the data only for one branch
        :returns: list of commits
        """
        query = """
        MATCH (b:Branch {{hash: "{0}"}})
        -[BranchCommit]->(c:Commit)
        return distinct c
        """.format(branch_hash)
        files_ = self.graph.run(query)
        return [x['c'] for x in files_.data()] 

    def get_files(self, branch_hash, project_id=None,
                  start_date=None, end_date=None):
        """Returns all files connected to a branch.
        Optionally it also filters by project_id
        :params branch_hash: branch unique identifier
        :params project_id: optional; if present the query
          returns the files from a specific project
        :param start_date: optional timestamp; filter files
          beginning with this date
        :param end_date: optional timestamp; filter files
          untill this date
        :returns: list of files
        """
        com_filter, where = fcid(project_id,
                                 start_date, end_date)
        fquery = """
        MATCH (b:Branch {{hash: "{0}"}})
              -[r:BranchCommit]->
              (c:Commit {1})
              -[UpdateFile]->
              (f: File)
        {2}
        RETURN collect(distinct f);
        """.format(branch_hash, com_filter, where)
        dt_ = self.graph.run(fquery)
        return [dict(x) for x in dt_.data()[0]['collect(distinct f)']]

    def get_files_updates(self, branch_hash, project_id=None,
                          start_date=None, end_date=None):
        """Returns all file update information (e.g. file complexity),
        for all files connected to a branch.
        Optionally it also filters by project_id
        :params branch_hash:  branch unique identifier
        :params project_id: optional; if present the query
          returns the files from a specific project
        :param start_date: optional timestamp; filter files
          beginning with this date
        :param end_date: optional timestamp; filter files
          untill this date
        :returns: list of file updates
        """
        com_filter, where = fcid(project_id,
                                 start_date, end_date)
        fuquery = """
        MATCH (b:Branch {{hash: "{0}"}})
              -[r:BranchCommit]->
              (c:Commit {1})
              -[fu: UpdateFile]->
              (f: File)
        {2}
        RETURN distinct fu;
        """.format(branch_hash, com_filter, where)

        dt_ = self.graph.run(fuquery)
        return [dict(x['fu']) for x in dt_.data()]

    def get_methods(self, branch_hash, project_id=None,
                    start_date=None, end_date=None):
        """Returns all methods connected to a branch.
        Optionally it also filters by project_id
        :params branch_hash:  branch unique identifier
        :params project_id: optional; if present the query
          returns the files from a specific project
        :param start_date: optional timestamp; filter files
          beginning with this date
        :param end_date: optional timestamp; filter files
          untill this date
        :returns: list of methods
        """
        com_filter, where = fcid(project_id,
                                 start_date, end_date)
        mquery = """
        MATCH (b:Branch {{hash: "{0}"}})
              -[r:BranchCommit]->
              (c:Commit {1})
              -[um: UpdateMethod]->
              (m: Method)
        {2}
        RETURN distinct m;
        """.format(branch_hash, com_filter, where)

        dt_ = self.graph.run(mquery)
        return [dict(x['m']) for x in dt_.data()]

    def get_method_updates(self, branch_hash, project_id=None,
                           start_date=None, end_date=None):
        """Returns all method update information, for all
        methods connected to a branch.
        Optionally it also filters by project_id
        :params branch_hash:  branch unique identifier
        :params project_id: optional; if present the query
          returns the files from a specific project
        :param start_date: optional timestamp; filter files
          beginning with this date
        :param end_date: optional timestamp; filter files
          untill this date
        :returns: list of method updates
        """
        com_filter, where = fcid(project_id,
                                 start_date, end_date)
        muquery = """
        MATCH (b:Branch {{hash: "{0}"}})
              -[r:BranchCommit]->
              (c:Commit {1})
              -[um: UpdateMethod]->
              ()
        {2}
        RETURN distinct um;
        """.format(branch_hash, com_filter, where)

        dt_ = self.graph.run(muquery)
        return [dict(x['um']) for x in dt_.data()]

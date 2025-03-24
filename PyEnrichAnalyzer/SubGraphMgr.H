#ifndef _SUBGRAPH_MGR
#define _SUBGRAPH_MGR

#include <vector>
#include <map>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

using namespace std;

class SubGraph;

class SubGraphMgr
{
	public:
		SubGraphMgr();
		~SubGraphMgr();
		int readSubGraphs(const char*);
		int initSubGraphsFromPython(const py::dict&, const py::str&);
		int readSubGraphAttributes(const char*);
		int readGeneList(const char*);
	        int initGeneListFromPython(const py::list&);
		int genRandomGraphs(int,int);
		int genRandomGraphs(int,int,vector<SubGraph*>&);
		vector<SubGraph*>& getSubGraphs();
		map<string,int>& getAttributeNames();
		SubGraph* getGraphAt(int);
		int getGeneCnt();
		map<string,int>& getGeneList();
		string& getSGName(int);
	private:
		vector<SubGraph*> subGraphSet;
		map<string,int> geneNameIDMap;
		map<int,string> geneIDNameMap;
		map<string,int> attributeNames;
		map<int,string> idNameMap;
		map<string, int> subGraphIdMap;

};
#endif

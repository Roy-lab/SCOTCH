#ifndef _SUBGRAPH
#define _SUBGRAPH
#include <map>
#include <vector>
#include <string>
using namespace std;
class SubGraph
{
	public:
		SubGraph();
		~SubGraph();
		int addVertex(const char*);
		map<string,int>& getVertexList();
		int setAttribute(double);
		vector<double>& getAttributes();
		bool hasSignificantAnnotation(double);
		bool hasSignificantAnnotation(double,map<int,string>&);
		bool isEnriched(int,double);
		inline map<string,double>& getEnrichments()
		{
			return enrichVals;
		}
	private:
		map<string,int> vertexList;
		vector<double> attributes;
		map<string,double> enrichVals;
};
#endif

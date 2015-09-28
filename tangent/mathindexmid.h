// (C) Copyright 2014 Andrew R. J. Kane <arkane@uwaterloo.ca>, All Rights Reserved.
//
//     Released for academic purposes only, All Other Rights Reserved.
//     This software is provided "as is" with no warranties, and the authors are not liable for any damages from its use.
//

// project: tangent-v3

// mathindexmid - classes and methods to implement some basic functionality of mathindex

#include "mathindexbase.h"

#define DEBUGCOUNTALLPOST false
#define DEBUGVERIFYPARSING false

//== Expression to Tuple Expansion ========================================================

unsigned windowvalue = 4;

string runl(string s) { int l=s.size(); if (l<6) { return s; } ostringstream ss; for (int i=0;;) { int e = s.find_first_not_of(s[i],i); ss<<((e<0?l:e)-i)<<s[i]; i=e; if (e<0) { return ss.str(); } } }

#define ENMAXCH 6
class TupleCB { public: virtual void tuple(string frte, string tote, string rel, string loc)=0; };
class TupleEmptyCB : public TupleCB { public: virtual void tuple(string frte, string tote, string rel, string loc) {} };
class TupleOutCB : public TupleCB { public: virtual void tuple(string frte, string tote, string rel, string loc) { cout<<"T\t"<<frte<<"\t"<<tote<<"\t"<<rel<<"\t"<<loc<<endl; } };
class ExprNode { public:
  char ty; string te; int chSize; ExprNode* ch[ENMAXCH]; // type, term, children (Note: children are owned)
  ExprNode(char type='-', string term="") { ty=type; te=term; chSize=0; for (int i=0; i<ENMAXCH; i++) { ch[i]=NULL; } }
  virtual ~ExprNode() { for (int i=0; i<chSize; i++) { delete ch[i]; ch[i]=NULL; } }
  bool children(char c) { for (int i=0; i<chSize; i++) { if (ch[i]->ty==c) return true; } return false; }
  void add(ExprNode* c) { if (c==NULL) return; if (chSize>=ENMAXCH) { cerr<<"WARNING: too many children adding "<<(*c)<<" to "<<(*this)<<endl; return; } ch[chSize++]=c; } // takes ownership
  void tuples(TupleCB& cb, int w, string loc, ExprNode* from, string rel) { rel+=ty; if (ty!='w' || te!="E!") { cb.tuple(from->te,te,rel,runl(loc)); } w--; if (w<=0) return; for (int i=0; i<chSize; i++) { ch[i]->tuples(cb,w,loc,from,rel); } }
  void tuples(TupleCB& cb, int w, string loc) { loc+=ty; for (int i=0; i<chSize; i++) { ch[i]->tuples(cb,w,loc,this,""); } if (!children('n') && te!="E!") { if (ENDOFBASELINE) { cb.tuple(te,"0!","n",runl(loc)); } } for (int i=0; i<chSize; i++) { ch[i]->tuples(cb,w,(ty=='-'?"":loc)); } }
  void prec(ostream& out) const { for (int i=0; i<chSize; i++) { out<<(*ch[i]); } } // recursive print of children
  friend ostream& operator<<(ostream& out, const ExprNode& t) { if (t.ty!='n' && t.ty!='-') { out<<","<<t.ty; } out<<"["<<t.te; t.prec(out); out<<"]"; return out; }
};

#define PEWARN(b,w) if (b) { ostringstream ss; ss<<"WARNING: " w " "<<s<<" at i="<<i; error=ss.str(); return; }

static void parseExprRec(string s, int& i, ExprNode* parent, string& error, char type='n') {
  if (i >= (int)s.size()) return; // end of recursion
  bool first = (i==0);
  PEWARN(s[i]!='[',"bad expression (0)"); i++;
  // first element - relation specified by input type
  int end = s.find_first_of("[,]",i); PEWARN(end<0 || end>=(int)s.size(),"bad expression reading past end"); // next control character
  char cc = s[end];
  PEWARN(end<=i,"bad expression (1)"); // must have value
  ExprNode* n = (first?parent:new ExprNode()); n->te=s.substr(i,end-i); if (!first) { n->ty=type; parent->add(n); parent=n; }
  switch (cc) {
    case '[': i=end; parseExprRec(s,i,parent,error); break;
    case ',': i=end+1; break;
    case ']': i=end+1; return;
    default: PEWARN(true,"bad expression (2)");
  }
  // remaining elements
  for (;;) { // value position indicates type is specified or not
    end = s.find_first_of("[,]",i); PEWARN(end<0 || end>=(int)s.size(),"bad expression reading past end"); // next control character
    char cc = s[end];
    switch (cc) {
      case '[': PEWARN(end!=i+1,"bad expression (3)"); type = s[i]; i=end; parseExprRec(s,i,parent,error,type); break; // other relation (single character type) followed by [...]
      case ',': PEWARN(end!=i,"bad expression (4)"); i++; break; // must not have a value
      case ']': PEWARN(end!=i,"bad expression (5)"); i++; return; // must not have a value
      default: PEWARN(true,"bad expression (6)");
    }
  }
}
static void parseExpr(TupleCB& cb, string s) {
  int i = 0; ExprNode root; string error=""; parseExprRec(s,i,&root,error); // parse
  if (error!="") cerr<<error<<" got "<<root<<endl; else { ostringstream ss; ss<<root; if (s!=ss.str()) cerr<<"WARNING: parse cannot reproduce input "<<s<<" "<<ss.str()<<endl; }
  root.tuples(cb,windowvalue,""); // expand
  if (DEBUGVERIFYPARSING) { stringstream ss; ss<<root; if (s!=ss.str()) { cerr<<"WARNING: '"<<s<<"' != '"<<ss.str()<<"'"<<endl; } }
}

//== Query/Output ========================================================

// TODO: can we remove all these virtual function calls?  maybe an array of IDIterPL pointers and keep them in sorted order...

#define IDEND INT_MAX

class IDIter {
public:
  int cv, cc; llong s, us; // value, count, size, unique size (no IND double counting)
  IDIter() { cv=-1; cc=-1; s=us=0; }
  virtual ~IDIter() {};
  virtual void skip(int v) = 0; // jump to >= v
  virtual void print(ostream& out) const = 0;
  virtual void printv(ostream& out, int v) const { if (cv==v) { print(out); } }
  friend ostream& operator<<(ostream& out, const IDIter& t) { t.print(out); return out; }
};

class IDIterEmpty : public IDIter { public:
  IDIterEmpty() { cv=IDEND; cc=0; }
  virtual void skip(int v) {}; // jump to >= v
  virtual void print(ostream& out) const { out<<"{empty}"; }
};

class IDIterPL : public IDIter {
  postingslist* pl; // not owned by this object
  llong& postsk;
  int ci; // index
  int qcount; // query count
  inline void getCurrent() { cv=pl->get(ci); cc=min(qcount,pl->getcount(ci)); rem=pl->getcount(ci)-cc; }
  inline void next() { ci++; if (ci>=s) { cv=IDEND; ci=s-1; } else { getCurrent(); } }
public:
  int rem; // remaining postings
  IDIterPL(postingslist* pl, int count, llong& p) : postsk(p) { this->pl=pl; s=us=pl->size(); ci=-1; qcount=count; rem=-1; next(); if (DEBUGCOUNTALLPOST) postsk++; } // start at first element
  virtual ~IDIterPL() { if (ci!=s-1) { cerr<<"INTERNAL ERROR: not at end of postingslist"<<endl; throw; } }
  virtual void skip(int v) {
    int ciOrig=ci;
    if (cv<v) { next(); } if (cv>=v) { if (DEBUGCOUNTALLPOST) postsk+=ci-ciOrig; return; } // could be v - 1...
    if (v>=IDEND) { cv=IDEND; ci=s-1; postsk+=ci-ciOrig; return; }
    // doubling search
    int jump=1;
    while (ci+jump < s && pl->get(ci+jump) < v) { jump*=2; }
    if (jump > 1) {
      jump /= 2;
      ci += jump;
      // binary search in range
      while (jump > 1) {
        jump /= 2;
        int split = ci+jump;
        if (split < s && pl->get(split) < v) { ci=split; }
      }
    }
    next();
    postsk+=ci-ciOrig;
  } // jump to >= v
  virtual void print(ostream& out) const { out<<"{"<<cv<<","<<cc<<"}"; }
};

class IDIterIND : public IDIter { // indirect iterator pointing to a shared IDIterPL
  IDIterPL* bs; // not owned, so do not delete
  inline void getCurrent() { cv=bs->cv; cc=min(1,bs->rem); bs->rem-=cc; }
public:
  IDIterIND(IDIterPL* base) { bs=base; s=bs->s; getCurrent(); }
  virtual void skip(int v) { if (bs->cv<v) { bs->skip(v); } getCurrent();  } // jump to >= v
  virtual void print(ostream& out) const { out<<"{IND,"<<*bs<<"}"; }
};

class IDIterOR : public IDIter { protected:
  string type;
  bool own; IDIter* x; IDIter* y;
  inline void getCurrent() { if (x->cv < y->cv) { cv=x->cv; cc=x->cc; } else if (x->cv > y->cv) { cv=y->cv; cc=y->cc; } else { cv=x->cv; cc=(x->cc)+(y->cc); } }
public:
  IDIterOR(IDIter* x, IDIter* y, bool own=true) { type="OR"; this->own=own; this->x=x; this->y=y; getCurrent(); s=x->s+y->s; us=x->us+y->us; }
  virtual ~IDIterOR() { if (own&&x!=NULL) { delete x; x=NULL; } if (own&&y!=NULL) { delete y; y=NULL; } };
  virtual void skip(int v) { if (x->cv<v) { x->skip(v); } if (y->cv<v) { y->skip(v); } getCurrent(); }; // jump to >= v
  virtual void print(ostream& out) const { out<<"["<<type<<(*x)<<","<<(*y)<<"]"; }
  virtual void printv(ostream& out, int v) const { if (cv==v) { out<<"["<<type<<"(cc="<<cc<<")"; x->printv(out,v); out<<","; y->printv(out,v); out<<"]"; } }
};

class IDIterADD : public IDIterOR { public: // LHS doesn generate values
  IDIterADD(IDIter* x, IDIter* y, bool own=true) : IDIterOR(x,y,own) { type="ADD"; }
  virtual void skip(int v) { if (x->cv<v) { x->skip(v); } if (y->cv<x->cv) { y->skip(x->cv); } getCurrent(); }; // jump to >= v
};

// TODO: mechanism not right for this when count > 1
class IDIterANY : public IDIterOR { // count always 1
  inline void getCurrent() { if (x->cv < y->cv) { cv=x->cv; cc=x->cc; } else if (x->cv > y->cv) { cv=y->cv; cc=y->cc; } else { cv=x->cv; cc=max(x->cc,y->cc); } }
public:
  IDIterANY(IDIter* x, IDIter* y, bool own=true) : IDIterOR(x,y,own) { type="ANY"; getCurrent(); }
  virtual void skip(int v) { if (x->cv<v) x->skip(v); if (x->cv==v&&x->cc==1&&v<IDEND) { cv=x->cv; cc=x->cc; return; } /* don't process right side if already at value */ if (y->cv<v) y->skip(v); getCurrent(); } // jump to >= v
  virtual void print(ostream& out) const { out<<"["<<type<<"...]"; } // don't expand IDIterANY
};


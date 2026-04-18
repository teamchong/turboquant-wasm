/**********************************************************
*      See the LICENSE file for copyright information.     *
**********************************************************/

#include "config.h"

#include<rbtree/red_black_tree.h>
#include<stdio.h>
#include<ctype.h>


/*  this file has functions to test a red-black tree of integers */

void IntDest(void* a) {
  free(a);
}



int IntComp(const void* a,const void* b) {
  if( *(int*)a > *(int*)b) return(1);
  if( *(int*)a < *(int*)b) return(-1);
  return(0);
}

int main() {
  int option=0;
  int newKey,newKey2;
  int* newInt;
  rb_red_blk_node* newNode;
  rb_red_blk_tree* tree;

  tree=RBTreeCreate(IntComp, IntDest);
  while (option != 6) {
    printf("choose one of the following:\n");
    printf("(1) add to tree\n(2) delete from tree\n(3) query\n");
    printf("(4) find predecessor\n(5) find successor\n");
    printf("(6) quit\n");
    do option=fgetc(stdin); while(-1 != option && isspace(option));
    option-='0';
    switch(option)
      {
      case 1:
	{
	  printf("type key for new node\n");
	  scanf("%i",&newKey);
	  newInt= malloc(sizeof(int));
	  *newInt=newKey;
	  RBTreeInsert(tree, newInt);
	}
	break;
	
      case 2:
	{
	  printf("type key of node to remove\n");
	  scanf("%i",&newKey);
	  if ( ( newNode=RBExactQuery(tree,&newKey ) ) ) RBDelete(tree,newNode);/*assignment*/
	  else printf("key not found in tree, no action taken\n");
	}
	break;

      case 3:
	{
	  printf("type key of node to query for\n");
	  scanf("%i",&newKey);
	  if ( ( newNode = RBExactQuery(tree,&newKey) ) ) {/*assignment*/
	    printf("data found in tree at location %i\n",(int)newNode);
	  } else {
	    printf("data not in tree\n");
	  }
	}
	break;
      case 4:
	{
	  printf("type key of node to find predecessor of\n");
	  scanf("%i",&newKey);
	  if ( ( newNode = RBExactQuery(tree,&newKey) ) ) {/*assignment*/
	    newNode=TreePredecessor(tree,newNode);
	    if(tree->nil == newNode) {
	      printf("there is no predecessor for that node (it is a minimum)\n");
	    } else {
	      printf("predecessor has key %i\n",*(int*)newNode->key);
	    }
	  } else {
	    printf("data not in tree\n");
	  }
	}
	break;
      case 5:
	{
	  printf("type key of node to find successor of\n");
	  scanf("%i",&newKey);
	  if ( (newNode = RBExactQuery(tree,&newKey) ) ) {
	    newNode=TreeSuccessor(tree,newNode);
	    if(tree->nil == newNode) {
	      printf("there is no successor for that node (it is a maximum)\n");
	    } else {
	      printf("successor has key %i\n",*(int*)newNode->key);
	    }
	  } else {
	    printf("data not in tree\n");
	  }
	}
	break;
      case 6:
	{
	  RBTreeDestroy(tree);
	  return 0;
	}
	break;
      default:
	printf("Invalid input; Please try again.\n");
      }
  }
  return 0;
}

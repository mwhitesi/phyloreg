

def getTreeDict( treefilename ):
	phylo = dict()
	
	parent = []
	stack = []
	
	with open( treefilename ) as treefile:
		while True:
			c = treefile.read(1)
			
			if not c:
				break
			
			
			#print ( c)
			#sys.exit()
		
			if c == ')':
				#print (stack )
				#sys.exit()
				top = ''
				child = ''
				while top != '(':
					top = stack.pop()
					child = child + top
	
				#print ( child )
				#sys.exit()
				child = child[::-1]
				#print ( child )
				child = child.replace('(', '')
				child = child.split(',')
				left = child[0].split(':')
				right = child[1].split(':')
				sp1 = left[0]
				sp1dist = left[1]
				sp2 = right[0]
				sp2dist = right[1]
	
				if sp1.isupper():
					firstname = sp1
				else:
					firstname = sp1[0].upper()
	
				if sp2.isupper():
					secondname = sp2
				else:
					secondname = sp2[0].upper()
	
				current_parent = firstname + secondname
							
				phylo[sp1] = ( current_parent, float(sp1dist) )
				phylo[sp2] = ( current_parent, float(sp2dist) )
	
				current_parent = current_parent[::-1]
	
				#print ( current_parent )
	
				stack.append( current_parent )
	
	
				#print ( phylo )
				#print ( stack )
				#sys.exit()
		
			else:
				stack.append(c)
	
	
	current_parent = current_parent[::-1]
	
	#print ( current_parent )
	
	phylo[ current_parent ] = ( None, None )

	#print (phylo )
	return phylo

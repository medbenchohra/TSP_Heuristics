import Lin_Keringhan_bakend as lkb
import random
#import TsplibParser
def Lin_Keringhan(cord,depth=3):
    neighbors= None
    verbose=False
    cities = []
    #instance=TsplibParser.load_instance(pbFileName,solFileName)
    #cord = instance.get_nodes_coord()
    for row in cord :
        cities.append(lkb.City(str(row[0]), int(row[1]), int(row[2])))
    random.shuffle(cities)
    lkb.Tour.init_roads(cities)
    tour = lkb.Tour(cities)

    tour, iteration = lkb.tour_improve(tour, neighbors, verbose, depth)

    return tour.length

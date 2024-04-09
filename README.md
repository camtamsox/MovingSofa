My calculus teacher introduced me to an unsolved math problem called the moving sofa problem. The problem asked for the largest area 2D sofa that can fit through an L-shaped hallway. Instead of using calculus to try to solve this problem, I decided to use a computational approach.
![sofa problem](https://github.com/camtamsox/MovingSofa/assets/109252429/264466d7-3f1d-4237-b1f7-c975973e3357)

_A sofa derived by John Hammersley. This animation is from Wikipedia._

I came up with an idea that consisted of two main parts: a generator to create new sofas (shapes represented by a set of vertices) and a path finder to figure out if the sofa fit through the hallway. The generator randomly selected and moved a vertex until the area of the new sofa was larger than the original. This new sofa would then be sent to the pathfinder to determine if the sofa could fit through the hallway. If it fit, the new sofa would be sent to generator, and if it didn't fit, the previous sofa would be sent to the generator. This process then repeated until the sofa's area reached a maximum.

![moving sofa diagram](https://github.com/camtamsox/MovingSofa/assets/109252429/eee72ce9-f040-4e4e-a62d-c084d0bdc0c8)

_The circular process between the generator and pathfinder that gradually increases the area of the sofa._

To create a pathfinder, I decided to frame the problem of finding a path into something that could be solved using a machine learning technique called reinforcement learning. This AI was given an input—the list of the initial vertex positions and the sofa’s current displacement/rotation—and had to output the direction it wanted the sofa to move/rotate (by a predetermined amount). This repeated until the sofa made it through the hallway or until the AI had tried so many times that it seemed unlikely that the sofa could fit. When the AI successfully figured out the sofa's path, it was rewarded, and if it didn't, it was penalized. These rewards and penalizations allowed the AI to, through trial and error, slowly learn how to fit different sofas through the hallway.

![moving sofa diagram2](https://github.com/camtamsox/MovingSofa/assets/109252429/4e42d776-54cf-4725-8945-386e77ac9197)

_The reinforcement learning AI that finds a path for the sofa through the hallway._

After running the program, it started to come up with some sofas. Interestingly, it had figured out how to cheat; some of the sofas had overlapping sections which had fooled a coding library I used (Shapely) into calculating an area that was too large. To combat this, I reordered the sofa's vertices by their distances from one another to try to eliminate the problematic overlapping sections. However, by taking advantage of my imperfect sorting algorithm, the program still managed to come up with sofas that had overlapping sections.

![1_72 area 20 verts](https://github.com/camtamsox/MovingSofa/assets/109252429/65390101-1400-4979-8406-5358b71d5dcd)

_This sofa had an incorrectly calculated area of 1.72._

When the program didn't cheat, it was only able to reach the local optimum of a rectangle. This could be because I had not generated enough sofas to give my program enough time to come up with a different solution. Also, this could be because the pathfinder wasn't able to find the paths for shapes that were more complex than a rectangle.


# Reference
https://golang.org/doc/tutorial/getting-started

# Fundamentals

```text
Go code is grouped into packages, and packages are grouped into modules. Your module specifies dependencies needed to run your code, including the Go version and the set of other modules it requires.

Go executes init functions automatically at program startup, after global variables have been initialized.
```

## Basic commands
```bash
# Enable dependency tracking
$ go mod init github.com/mymodule
# run your code
$ cd hello
$ go run .
# Add new module requirements and sums.
# go.sum is used in authenticating the module.
$ go mod tidy
# Edit module to use local module
$ go mod edit -replace example.com/greetings=../greetings
# result will be like: replace example.com/greetings => ../greetings
# test
$ go test
$ go test -v
# build
$ go build
# discover the install path
$ go list -f '{{.Target}}'
# change install target
$ go env -w GOBIN=/path/to/your/bin
# compile and install the package
$ go install
```

## Basic constructs
```go
// variables
// The var statement declares a list of variables; as in function argument lists, the type is last.
// A var statement can be at package or function level. We see both in this example.
var c, python, java bool

func main() {
	var i int
	fmt.Println(i, c, python, java)
}

// A var declaration can include initializers, one per variable.
var i, j int = 1, 2
// If an initializer is present, the type can be omitted; the variable will take the type of the initializer.
var c, python, java = true, false, "no!"

// Short variable declarations
// Inside a function, the := short assignment statement can be used in place of a var declaration with implicit type.
c, python, java := true, false, "no!"
// Outside a function, every statement begins with a keyword (var, func, and so on) and so the := construct is not available.

// In Go, the := operator is a shortcut for declaring and initializing a variable in one line (Go uses the value on the right to determine the variable's type).
message := fmt.Sprintf("Hi, %v. Welcome!", name)
// the following is the same
var message string
message = fmt.Sprintf("Hi, %v. Welcome!", name)

// Basic types
bool
string
int  int8  int16  int32  int64
uint uint8 uint16 uint32 uint64 uintptr
byte // alias for uint8
rune // alias for int32
     // represents a Unicode code point
float32 float64
complex64 complex128

// Type conversion
// The expression T(v) converts the value v to the type T.
i := 42
f := float64(i)
u := uint(f)
// Unlike in C, in Go assignment between items of different type requires an explicit conversion.

// Constants
// Constants are declared like variables, but with the const keyword.
// Constants cannot be declared using the := syntax.
const Pi = 3.14
const World = "世界"

// Function definition
func Hello(name string) string
// This function takes a name parameter whose type is string. The function also returns a string. In Go, a function whose name starts with a capital letter can be called by a function not in the same package. This is known in Go as an exported name.

func add(x, y int) int
// is the same as
func add(x int, y int) int

// Named return values
func split(sum int) (x, y int) {
	x = sum * 4 / 9
	y = sum - x
	return // "naked" return, should only be used in short functions, can harm readability in longer functions.
}

// map
// A map to associate names with messages.
messages := make(map[string]string)

// for loop
// Basic for loop
func main() {
	sum := 0
	for i := 0; i < 10; i++ { // braces always required
		sum += i
	}
	fmt.Println(sum)
}
// The init and post statements are optional.
func main() {
	sum := 1
	for ; sum < 1000; {
		sum += sum
	}
	fmt.Println(sum)
}
// go's "while" loop
func main() {
	sum := 1
	for sum < 1000 {
		sum += sum
	}
	fmt.Println(sum)
}
// forever
func main() {
	for {
	}
}
// Loop through the received slice of names, calling
// the Hello function to get a message for each name. names here is a []string
for _, name := range names {
    message, err := Hello(name)
    if err != nil {
        return nil, err
    }
    // In the map, associate the retrieved message with
    // the name.
    messages[name] = message
}

// If
func sqrt(x float64) string {
	if x < 0 {
		return sqrt(-x) + "i"
	}
	return fmt.Sprint(math.Sqrt(x))
}
// the if statement can start with a short statement to execute before the condition.
// Variables declared by the statement are only in scope until the end of the if.
func pow(x, n, lim float64) float64 {
	if v := math.Pow(x, n); v < lim {
		return v
	}
	return lim
}

// Switch
// no need to 'break'
// Switch cases evaluate cases from top to bottom, stopping when a case succeeds.
func main() {
	fmt.Print("Go runs on ")
	switch os := runtime.GOOS; os {
	case "darwin":
		fmt.Println("OS X.")
	case "linux":
		fmt.Println("Linux.")
	default:
		// freebsd, openbsd,
		// plan9, windows...
		fmt.Printf("%s.\n", os)
	}
}
// Switch without a condition is the same as switch true.
func main() {
	t := time.Now()
	switch {
	case t.Hour() < 12:
		fmt.Println("Good morning!")
	case t.Hour() < 17:
		fmt.Println("Good afternoon.")
	default:
		fmt.Println("Good evening.")
	}
}

// Defer
// A defer statement defers the execution of a function until the surrounding function returns.
// The deferred call's arguments are evaluated immediately, but the function call is not executed until the surrounding function returns.
func main() {
	defer fmt.Println("world")

	fmt.Println("hello")
}
//Result:
// hello
// world

// Stacking defers
// Deferred function calls are pushed onto a stack. When a function returns, its deferred calls are executed in last-in-first-out order.
func main() {
	fmt.Println("counting")

	for i := 0; i < 10; i++ {
		defer fmt.Println(i)
	}

	fmt.Println("done")
}
//Result:
// counting
// done
// 9
// 8
// ...
// 0

// Pointers
// Go has pointers. A pointer holds the memory address of a value.
// The type *T is a pointer to a T value. Its zero value is nil.
var p *int
// The & operator generates a pointer to its operand.
i := 42
p = &i
// The * operator denotes the pointer's underlying value.
fmt.Println(*p) // read i through the pointer p
*p = 21         // set i through the pointer p
// This is known as "dereferencing" or "indirecting".
// Unlike C, Go has no pointer arithmetic.

// Structs
// A struct is a collection of fields.
type Vertex struct {
	X int
	Y int
}
// Struct fields are accessed using a dot.
func main() {
	v := Vertex{1, 2}
	v.X = 4
	fmt.Println(v.X)
}
// Struct fields can be accessed through a struct pointer.
// To access the field X of a struct when we have the struct pointer p we could write (*p).X. However, that notation is cumbersome, so the language permits us instead to write just p.X, without the explicit dereference.
func main() {
	v := Vertex{1, 2}
	p := &v
	p.X = 1e9
	fmt.Println(v)
}

// Struct Literals
// A struct literal denotes a newly allocated struct value by listing the values of its fields.
// You can list just a subset of fields by using the Name: syntax. (And the order of named fields is irrelevant.)
// The special prefix & returns a pointer to the struct value.
type Vertex struct {
	X, Y int
}

var (
	v1 = Vertex{1, 2}  // has type Vertex
	v2 = Vertex{X: 1}  // Y:0 is implicit
	v3 = Vertex{}      // X:0 and Y:0
	p  = &Vertex{1, 2} // has type *Vertex
)

func main() {
	fmt.Println(v1, p, v2, v3)
}

// Arrays
// The type [n]T is an array of n values of type T.
var a [10]int // declares a variable a as an array of ten integers.
// An array's length is part of its type, so arrays cannot be resized. This seems limiting, but don't worry; Go provides a convenient way of working with arrays.
func main() {
	var a [2]string
	a[0] = "Hello"
	a[1] = "World"
	fmt.Println(a[0], a[1])
	fmt.Println(a)

	primes := [6]int{2, 3, 5, 7, 11, 13}
	fmt.Println(primes)
}

// Slices
//An array has a fixed size. A slice, on the other hand, is a dynamically-sized, flexible view into the elements of an array. In practice, slices are much more common than arrays.
//The type []T is a slice with elements of type T.
//A slice is formed by specifying two indices, a low and high bound, separated by a colon:
a[low : high]
// This selects a half-open range which includes the first element, but excludes the last one.
// The following expression creates a slice which includes elements 1 through 3 of a:
a[1:4]

func main() {
	primes := [6]int{2, 3, 5, 7, 11, 13}

	var s []int = primes[1:4]
	fmt.Println(s)
}

// Slices are like references to arrays
// A slice does not store any data, it just describes a section of an underlying array.
// Changing the elements of a slice modifies the corresponding elements of its underlying array.
// Other slices that share the same underlying array will see those changes.

// Slice literals
// A slice literal is like an array literal without the length.
[3]bool{true, true, false} // This is an array literal
[]bool{true, true, false} // And this creates the same array as above, then builds a slice that references it
func main() {
	q := []int{2, 3, 5, 7, 11, 13}
	fmt.Println(q)

	r := []bool{true, false, true, true, false, true}
	fmt.Println(r)

	s := []struct {
		i int
		b bool
	}{
		{2, true},
		{3, false},
		{5, true},
		{7, true},
		{11, false},
		{13, true},
	}
	fmt.Println(s)
}

// Slice length and capacity
// A slice has both a length and a capacity.
// The length of a slice is the number of elements it contains.
// The capacity of a slice is the number of elements in the underlying array, counting from the first element in the slice.
// The length and capacity of a slice s can be obtained using the expressions len(s) and cap(s).
// You can extend a slice's length by re-slicing it, provided it has sufficient capacity. Try changing one of the slice operations in the example program to extend it beyond its capacity and see what happens.
func main() {
	s := []int{2, 3, 5, 7, 11, 13}
	printSlice(s)

	// Slice the slice to give it zero length.
	s = s[:0]
	printSlice(s)

	// Extend its length.
	s = s[:4]
	printSlice(s)

	// Drop its first two values.
	s = s[2:]
	printSlice(s)
}

func printSlice(s []int) {
	fmt.Printf("len=%d cap=%d %v\n", len(s), cap(s), s)
}
/* outputs:
len=6 cap=6 [2 3 5 7 11 13]
len=0 cap=6 []
len=4 cap=6 [2 3 5 7]
len=2 cap=4 [5 7]
*/
// The zero value of a slice is nil. A nil slice has a length and capacity of 0 and has no underlying array.

// Creating a slice with make
// Slices can be created with the built-in make function; this is how you create dynamically-sized arrays.
// The make function allocates a zeroed array and returns a slice that refers to that array:
a := make([]int, 5)  // len(a)=5
// To specify a capacity, pass a third argument to make:
b := make([]int, 0, 5) // len(b)=0, cap(b)=5
b = b[:cap(b)] // len(b)=5, cap(b)=5
b = b[1:]      // len(b)=4, cap(b)=4
// Slices can contain any type, including other slices.
func main() {
	// Create a tic-tac-toe board.
	board := [][]string{
		[]string{"_", "_", "_"},
		[]string{"_", "_", "_"},
		[]string{"_", "_", "_"},
	}

	// The players take turns.
	board[0][0] = "X"
	board[2][2] = "O"
	board[1][2] = "X"
	board[1][0] = "O"
	board[0][2] = "X"

	for i := 0; i < len(board); i++ {
		fmt.Printf("%s\n", strings.Join(board[i], " "))
	}
}
// Appending to a slice
// It is common to append new elements to a slice, and so Go provides a built-in append function. The documentation of the built-in package describes append.
func append(s []T, vs ...T) []T
// The first parameter s of append is a slice of type T, and the rest are T values to append to the slice.
// The resulting value of append is a slice containing all the elements of the original slice plus the provided values.
// If the backing array of s is too small to fit all the given values a bigger array will be allocated. The returned slice will point to the newly allocated array.*/
func main() {
	var s []int
	printSlice(s)

	// append works on nil slices.
	s = append(s, 0)
	printSlice(s)

	// The slice grows as needed.
	s = append(s, 1)
	printSlice(s)

	// We can add more than one element at a time.
	s = append(s, 2, 3, 4)
	printSlice(s)
}

func printSlice(s []int) {
	fmt.Printf("len=%d cap=%d %v\n", len(s), cap(s), s)
}
/* output:
len=0 cap=0 []
len=1 cap=1 [0]
len=2 cap=2 [0 1]
len=5 cap=6 [0 1 2 3 4]
*/

// A slice is like an array, except that its size changes dynamically as you add and remove items.
names := []string{"Gladys", "Samantha", "Darrin"}

// Range
// The range form of the for loop iterates over a slice or map.
// When ranging over a slice, two values are returned for each iteration. The first is the index, and the second is a copy of the element at that index.
var pow = []int{1, 2, 4, 8, 16, 32, 64, 128}

func main() {
	for i, v := range pow {
		fmt.Printf("2**%d = %d\n", i, v)
	}
}
// You can skip the index or value by assigning to _.
for i, _ := range pow
for _, value := range pow
// If you only want the index, you can omit the second variable.
for i := range pow

// Maps
// A map maps keys to values.
// The zero value of a map is nil. A nil map has no keys, nor can keys be added.
// The make function returns a map of the given type, initialized and ready for use.
type Vertex struct {
	Lat, Long float64
}

var m map[string]Vertex

func main() {
	m = make(map[string]Vertex)
	m["Bell Labs"] = Vertex{
		40.68433, -74.39967,
	}
	fmt.Println(m["Bell Labs"])
}
// Map literals are like struct literals, but the keys are required.
type Vertex struct {
	Lat, Long float64
}

var m = map[string]Vertex{
	"Bell Labs": Vertex{
		40.68433, -74.39967,
	},
	"Google": Vertex{
		37.42202, -122.08408,
	},
}

func main() {
	fmt.Println(m)
}
// If the top-level type is just a type name, you can omit it from the elements of the literal.
var m = map[string]Vertex{
	"Bell Labs": {40.68433, -74.39967},
	"Google":    {37.42202, -122.08408},
}
// Mutating Maps
// Insert or update an element in map m:
m[key] = elem
// Retrieve an element:
elem = m[key]
// Delete an element:
delete(m, key)
// Test that a key is present with a two-value assignment:
elem, ok = m[key]
// If key is in m, ok is true. If not, ok is false.
// If key is not in the map, then elem is the zero value for the map's element type.
// Note: If elem or ok have not yet been declared you could use a short declaration form:
elem, ok := m[key]

// Function values
// Functions are values too. They can be passed around just like other values.
// Function values may be used as function arguments and return values.
func compute(fn func(float64, float64) float64) float64 {
	return fn(3, 4)
}

func main() {
	hypot := func(x, y float64) float64 {
		return math.Sqrt(x*x + y*y)
	}
	fmt.Println(hypot(5, 12))

	fmt.Println(compute(hypot))
	fmt.Println(compute(math.Pow))
}

// Function closures
// Go functions may be closures. A closure is a function value that references variables from outside its body. The function may access and assign to the referenced variables; in this sense the function is "bound" to the variables.
// For example, the adder function returns a closure. Each closure is bound to its own sum variable.
func adder() func(int) int {
	sum := 0
	return func(x int) int {
		sum += x
		return sum
	}
}

func main() {
	pos, neg := adder(), adder()
	for i := 0; i < 10; i++ {
		fmt.Println(
			pos(i),
			neg(-2*i),
		)
	}
}

// Methods
// Go does not have classes. However, you can define methods on types.
// A method is a function with a special receiver argument.
// The receiver appears in its own argument list between the func keyword and the method name.
// In this example, the Abs method has a receiver of type Vertex named v.
type Vertex struct {
	X, Y float64
}

func (v Vertex) Abs() float64 {
	return math.Sqrt(v.X*v.X + v.Y*v.Y)
}

func main() {
	v := Vertex{3, 4}
	fmt.Println(v.Abs())
}
// You can declare a method on non-struct types, too.
// In this example we see a numeric type MyFloat with an Abs method.
// You can only declare a method with a receiver whose type is defined in the same package as the method. You cannot declare a method with a receiver whose type is defined in another package (which includes the built-in types such as int).
type MyFloat float64

func (f MyFloat) Abs() float64 {
	if f < 0 {
		return float64(-f)
	}
	return float64(f)
}

func main() {
	f := MyFloat(-math.Sqrt2)
	fmt.Println(f.Abs())
}

// Pointer receivers
// You can declare methods with pointer receivers.
// This means the receiver type has the literal syntax *T for some type T. (Also, T cannot itself be a pointer such as *int.)
// For example, the Scale method here is defined on *Vertex.
// Methods with pointer receivers can modify the value to which the receiver points (as Scale does here). Since methods often need to modify their receiver, pointer receivers are more common than value receivers.
// Try removing the * from the declaration of the Scale function on line 16 and observe how the program's behavior changes.
//With a value receiver, the Scale method operates on a copy of the original Vertex value. (This is the same behavior as for any other function argument.) The Scale method must have a pointer receiver to change the Vertex value declared in the main function.
type Vertex struct {
	X, Y float64
}

func (v Vertex) Abs() float64 {
	return math.Sqrt(v.X*v.X + v.Y*v.Y)
}

func (v *Vertex) Scale(f float64) {
	v.X = v.X * f
	v.Y = v.Y * f
}

func main() {
	v := Vertex{3, 4}
	v.Scale(10)
	fmt.Println(v.Abs())
}

// Methods and pointer indirection
// Comparing the previous two programs, you might notice that functions with a pointer argument must take a pointer:
var v Vertex
ScaleFunc(v, 5)  // Compile error!
ScaleFunc(&v, 5) // OK
// while methods with pointer receivers take either a value or a pointer as the receiver when they are called:
var v Vertex
v.Scale(5)  // OK
p := &v
p.Scale(10) // OK
// For the statement v.Scale(5), even though v is a value and not a pointer, the method with the pointer receiver is called automatically. That is, as a convenience, Go interprets the statement v.Scale(5) as (&v).Scale(5) since the Scale method has a pointer receiver.
type Vertex struct {
	X, Y float64
}

func (v *Vertex) Scale(f float64) {
	v.X = v.X * f
	v.Y = v.Y * f
}

func ScaleFunc(v *Vertex, f float64) {
	v.X = v.X * f
	v.Y = v.Y * f
}

func main() {
	v := Vertex{3, 4}
	v.Scale(2)
	ScaleFunc(&v, 10)

	p := &Vertex{4, 3}
	p.Scale(3)
	ScaleFunc(p, 8)

	fmt.Println(v, p)
}
// output: {60 80} &{96 72}
// The equivalent thing happens in the reverse direction.
// Functions that take a value argument must take a value of that specific type:
var v Vertex
fmt.Println(AbsFunc(v))  // OK
fmt.Println(AbsFunc(&v)) // Compile error!
// while methods with value receivers take either a value or a pointer as the receiver when they are called:
var v Vertex
fmt.Println(v.Abs()) // OK
p := &v
fmt.Println(p.Abs()) // OK
// In this case, the method call p.Abs() is interpreted as (*p).Abs().
type Vertex struct {
	X, Y float64
}

func (v Vertex) Abs() float64 {
	return math.Sqrt(v.X*v.X + v.Y*v.Y)
}

func AbsFunc(v Vertex) float64 {
	return math.Sqrt(v.X*v.X + v.Y*v.Y)
}

func main() {
	v := Vertex{3, 4}
	fmt.Println(v.Abs())
	fmt.Println(AbsFunc(v))

	p := &Vertex{4, 3}
	fmt.Println(p.Abs())
	fmt.Println(AbsFunc(*p))
}

// Choosing a value or pointer receiver
// There are two reasons to use a pointer receiver.
// The first is so that the method can modify the value that its receiver points to.
// The second is to avoid copying the value on each method call. This can be more efficient if the receiver is a large struct, for example.
// In this example, both Scale and Abs are with receiver type *Vertex, even though the Abs method needn't modify its receiver.
// In general, all methods on a given type should have either value or pointer receivers, but not a mixture of both. (We'll see why over the next few pages.)
type Vertex struct {
	X, Y float64
}

func (v *Vertex) Scale(f float64) {
	v.X = v.X * f
	v.Y = v.Y * f
}

func (v *Vertex) Abs() float64 {
	return math.Sqrt(v.X*v.X + v.Y*v.Y)
}

func main() {
	v := &Vertex{3, 4}
	fmt.Printf("Before scaling: %+v, Abs: %v\n", v, v.Abs())
	v.Scale(5)
	fmt.Printf("After scaling: %+v, Abs: %v\n", v, v.Abs())
}

// Interfaces
// An interface type is defined as a set of method signatures.
// A value of interface type can hold any value that implements those methods.
// Note: There is an error in the example code on line 22. Vertex (the value type) doesn't implement Abser because the Abs method is defined only on *Vertex (the pointer type).
type Abser interface {
	Abs() float64
}

func main() {
	var a Abser
	f := MyFloat(-math.Sqrt2)
	v := Vertex{3, 4}

	a = f  // a MyFloat implements Abser
	a = &v // a *Vertex implements Abser

	// In the following line, v is a Vertex (not *Vertex)
	// and does NOT implement Abser.
	a = v

	fmt.Println(a.Abs())
}

type MyFloat float64

func (f MyFloat) Abs() float64 {
	if f < 0 {
		return float64(-f)
	}
	return float64(f)
}

type Vertex struct {
	X, Y float64
}

func (v *Vertex) Abs() float64 {
	return math.Sqrt(v.X*v.X + v.Y*v.Y)
}

// Interfaces are implemented implicitly
// A type implements an interface by implementing its methods. There is no explicit declaration of intent, no "implements" keyword.
// Implicit interfaces decouple the definition of an interface from its implementation, which could then appear in any package without prearrangement.
type I interface {
	M()
}

type T struct {
	S string
}

// This method means type T implements the interface I,
// but we don't need to explicitly declare that it does so.
func (t T) M() {
	fmt.Println(t.S)
}

func main() {
	var i I = T{"hello"}
	i.M()
}

// Interface values
// Under the hood, interface values can be thought of as a tuple of a value and a concrete type:
// (value, type)
// An interface value holds a value of a specific underlying concrete type.
// Calling a method on an interface value executes the method of the same name on its underlying type.

// Interface values with nil underlying values
// If the concrete value inside the interface itself is nil, the method will be called with a nil receiver.
// In some languages this would trigger a null pointer exception, but in Go it is common to write methods that gracefully handle being called with a nil receiver (as with the method M in this example.)
// Note that an interface value that holds a nil concrete value is itself non-nil.
type I interface {
	M()
}

type T struct {
	S string
}

func (t *T) M() {
	if t == nil {
		fmt.Println("<nil>")
		return
	}
	fmt.Println(t.S)
}

func main() {
	var i I

	var t *T
	i = t
	describe(i)
	i.M()

	i = &T{"hello"}
	describe(i)
	i.M()
}

func describe(i I) {
	fmt.Printf("(%v, %T)\n", i, i)
}
/* output
(<nil>, *main.T)
<nil>
(&{hello}, *main.T)
hello
*/

// Nil interface values
// A nil interface value holds neither value nor concrete type.
// Calling a method on a nil interface is a run-time error because there is no type inside the interface tuple to indicate which concrete method to call.

// The empty interface
// The interface type that specifies zero methods is known as the empty interface:
interface{}
// An empty interface may hold values of any type. (Every type implements at least zero methods.)
// Empty interfaces are used by code that handles values of unknown type. For example, fmt.Print takes any number of arguments of type interface{}.
func main() {
	var i interface{}
	describe(i)

	i = 42
	describe(i)

	i = "hello"
	describe(i)
}

func describe(i interface{}) {
	fmt.Printf("(%v, %T)\n", i, i)
}
/* output
(<nil>, <nil>)
(42, int)
(hello, string)
*/

// Type assertions
// A type assertion provides access to an interface value's underlying concrete value.
t := i.(T)
//This statement asserts that the interface value i holds the concrete type T and assigns the underlying T value to the variable t.
// If i does not hold a T, the statement will trigger a panic.
// To test whether an interface value holds a specific type, a type assertion can return two values: the underlying value and a boolean value that reports whether the assertion succeeded.
t, ok := i.(T)
// If i holds a T, then t will be the underlying value and ok will be true.
// If not, ok will be false and t will be the zero value of type T, and no panic occurs.
// Note the similarity between this syntax and that of reading from a map.
func main() {
	var i interface{} = "hello"

	s := i.(string)
	fmt.Println(s)

	s, ok := i.(string)
	fmt.Println(s, ok)

	f, ok := i.(float64)
	fmt.Println(f, ok)

	f = i.(float64) // panic
	fmt.Println(f)
}

// Type switches
// A type switch is a construct that permits several type assertions in series.
// A type switch is like a regular switch statement, but the cases in a type switch specify types (not values), and those values are compared against the type of the value held by the given interface value.
switch v := i.(type) {
case T:
    // here v has type T
case S:
    // here v has type S
default:
    // no match; here v has the same type as i
}
// The declaration in a type switch has the same syntax as a type assertion i.(T), but the specific type T is replaced with the keyword type.
// This switch statement tests whether the interface value i holds a value of type T or S. In each of the T and S cases, the variable v will be of type T or S respectively and hold the value held by i. In the default case (where there is no match), the variable v is of the same interface type and value as i.
func do(i interface{}) {
	switch v := i.(type) {
	case int:
		fmt.Printf("Twice %v is %v\n", v, v*2)
	case string:
		fmt.Printf("%q is %v bytes long\n", v, len(v))
	default:
		fmt.Printf("I don't know about type %T!\n", v)
	}
}

func main() {
	do(21)
	do("hello")
	do(true)
}
/* output
Twice 21 is 42
"hello" is 5 bytes long
I don't know about type bool!
*/

// Stringers
// One of the most ubiquitous interfaces is Stringer defined by the fmt package.
type Stringer interface {
    String() string
}
// A Stringer is a type that can describe itself as a string. The fmt package (and many others) look for this interface to print values.
type Person struct {
	Name string
	Age  int
}

func (p Person) String() string {
	return fmt.Sprintf("%v (%v years)", p.Name, p.Age)
}

func main() {
	a := Person{"Arthur Dent", 42}
	z := Person{"Zaphod Beeblebrox", 9001}
	fmt.Println(a, z)
}

// Errors
// Go programs express error state with error values.
// The error type is a built-in interface similar to fmt.Stringer:
type error interface {
    Error() string
}
// (As with fmt.Stringer, the fmt package looks for the error interface when printing values.)
// Functions often return an error value, and calling code should handle errors by testing whether the error equals nil.
i, err := strconv.Atoi("42")
if err != nil {
    fmt.Printf("couldn't convert number: %v\n", err)
    return
}
fmt.Println("Converted integer:", i)
// A nil error denotes success; a non-nil error denotes failure.

// Readers
// The io package specifies the io.Reader interface, which represents the read end of a stream of data.
// The Go standard library contains many implementations of this interface, including files, network connections, compressors, ciphers, and others.
// The io.Reader interface has a Read method:
func (T) Read(b []byte) (n int, err error)
// Read populates the given byte slice with data and returns the number of bytes populated and an error value. It returns an io.EOF error when the stream ends.
// The example code creates a strings.Reader and consumes its output 8 bytes at a time.
func main() {
	r := strings.NewReader("Hello, Reader!")

	b := make([]byte, 8)
	for {
		n, err := r.Read(b)
		fmt.Printf("n = %v err = %v b = %v\n", n, err, b)
		fmt.Printf("b[:n] = %q\n", b[:n])
		if err == io.EOF {
			break
		}
	}
}

// Goroutines
// A goroutine is a lightweight thread managed by the Go runtime.
go f(x, y, z)
// starts a new goroutine running
f(x, y, z)
// The evaluation of f, x, y, and z happens in the current goroutine and the execution of f happens in the new goroutine.
// Goroutines run in the same address space, so access to shared memory must be synchronized. The sync package provides useful primitives, although you won't need them much in Go as there are other primitives. (See the next slide.)
func say(s string) {
	for i := 0; i < 5; i++ {
		time.Sleep(100 * time.Millisecond)
		fmt.Println(s)
	}
}

func main() {
	go say("world")
	say("hello")
}

// Channels
// Channels are a typed conduit through which you can send and receive values with the channel operator, <-.
ch <- v    // Send v to channel ch.
v := <-ch  // Receive from ch, and
           // assign value to v.
// (The data flows in the direction of the arrow.)
// Like maps and slices, channels must be created before use:
// ch := make(chan int)
// By default, sends and receives block until the other side is ready. This allows goroutines to synchronize without explicit locks or condition variables.
// The example code sums the numbers in a slice, distributing the work between two goroutines. Once both goroutines have completed their computation, it calculates the final result.
func sum(s []int, c chan int) {
	sum := 0
	for _, v := range s {
		sum += v
	}
	c <- sum // send sum to c
}

func main() {
	s := []int{7, 2, 8, -9, 4, 0}

	c := make(chan int)
	go sum(s[:len(s)/2], c)
	go sum(s[len(s)/2:], c)
	x, y := <-c, <-c // receive from c

	fmt.Println(x, y, x+y)
}
/* output
-5 17 12
*/

// Buffered Channels
// Channels can be buffered. Provide the buffer length as the second argument to make to initialize a buffered channel:
ch := make(chan int, 100)
// Sends to a buffered channel block only when the buffer is full. Receives block when the buffer is empty.
func main() {
	ch := make(chan int, 2)
	ch <- 1
	ch <- 2
	fmt.Println(<-ch)
	fmt.Println(<-ch)
}

// Range and Close
// A sender can close a channel to indicate that no more values will be sent. Receivers can test whether a channel has been closed by assigning a second parameter to the receive expression: after
v, ok := <-ch
// ok is false if there are no more values to receive and the channel is closed.
// The loop for i := range c receives values from the channel repeatedly until it is closed.
// Note: Only the sender should close a channel, never the receiver. Sending on a closed channel will cause a panic.
// Another note: Channels aren't like files; you don't usually need to close them. Closing is only necessary when the receiver must be told there are no more values coming, such as to terminate a range loop.
func fibonacci(n int, c chan int) {
	x, y := 0, 1
	for i := 0; i < n; i++ {
		c <- x
		x, y = y, x+y
	}
	close(c)
}

func main() {
	c := make(chan int, 10)
	go fibonacci(cap(c), c)
	for i := range c {
		fmt.Println(i)
	}
}

// Select
// The select statement lets a goroutine wait on multiple communication operations.
// A select blocks until one of its cases can run, then it executes that case. It chooses one at random if multiple are ready.
func fibonacci(c, quit chan int) {
	x, y := 0, 1
	for {
		select {
		case c <- x:
			x, y = y, x+y
		case <-quit:
			fmt.Println("quit")
			return
		}
	}
}

func main() {
	c := make(chan int)
	quit := make(chan int)
	go func() {
		for i := 0; i < 10; i++ {
			fmt.Println(<-c)
		}
		quit <- 0
	}()
	fibonacci(c, quit)
}
// Default Selection
// The default case in a select is run if no other case is ready.
// Use a default case to try a send or receive without blocking:
select {
case i := <-c:
    // use i
default:
    // receiving from c would block
}
func main() {
	tick := time.Tick(100 * time.Millisecond)
	boom := time.After(500 * time.Millisecond)
	for {
		select {
		case <-tick:
			fmt.Println("tick.")
		case <-boom:
			fmt.Println("BOOM!")
			return
		default:
			fmt.Println("    .")
			time.Sleep(50 * time.Millisecond)
		}
	}
}
```

### Error handling
```go
package greetings

import (
    "errors"
    "fmt"
)

// Hello returns a greeting for the named person.
func Hello(name string) (string, error) {
    // If no name was given, return an error with a message.
    if name == "" {
        return "", errors.New("empty name")
    }

    // If a name was received, return a value that embeds the name
    // in a greeting message.
    message := fmt.Sprintf("Hi, %v. Welcome!", name)
    return message, nil
}

func main() {
    // Set properties of the predefined Logger, including
    // the log entry prefix and a flag to disable printing
    // the time, source file, and line number.
    log.SetPrefix("greetings: ")
    log.SetFlags(0)

    // Request a greeting message.
    message, err := greetings.Hello("")
    // If an error was returned, print it to the console and
    // exit the program.
    if err != nil {
        log.Fatal(err)
    }

    // If no error was returned, print the returned message
    // to the console.
    fmt.Println(message)
}

// sync.Mutex
// We've seen how channels are great for communication among goroutines.
// But what if we don't need communication? What if we just want to make sure only one goroutine can access a variable at a time to avoid conflicts?
// This concept is called mutual exclusion, and the conventional name for the data structure that provides it is mutex.
// Go's standard library provides mutual exclusion with sync.Mutex and its two methods:
Lock
Unlock
// We can define a block of code to be executed in mutual exclusion by surrounding it with a call to Lock and Unlock as shown on the Inc method.
// We can also use defer to ensure the mutex will be unlocked as in the Value method.
// SafeCounter is safe to use concurrently.
type SafeCounter struct {
	mu sync.Mutex
	v  map[string]int
}

// Inc increments the counter for the given key.
func (c *SafeCounter) Inc(key string) {
	c.mu.Lock()
	// Lock so only one goroutine at a time can access the map c.v.
	c.v[key]++
	c.mu.Unlock()
}

// Value returns the current value of the counter for the given key.
func (c *SafeCounter) Value(key string) int {
	c.mu.Lock()
	// Lock so only one goroutine at a time can access the map c.v.
	defer c.mu.Unlock()
	return c.v[key]
}

func main() {
	c := SafeCounter{v: make(map[string]int)}
	for i := 0; i < 1000; i++ {
		go c.Inc("somekey")
	}

	time.Sleep(time.Second)
	fmt.Println(c.Value("somekey"))
}
```

### Testing
```go
package greetings

import (
    "testing"
    "regexp"
)

// TestHelloName calls greetings.Hello with a name, checking
// for a valid return value.
func TestHelloName(t *testing.T) {
    name := "Gladys"
    want := regexp.MustCompile(`\b`+name+`\b`)
    msg, err := Hello("Gladys")
    if !want.MatchString(msg) || err != nil {
        t.Fatalf(`Hello("Gladys") = %q, %v, want match for %#q, nil`, msg, err, want)
    }
}

// TestHelloEmpty calls greetings.Hello with an empty string,
// checking for an error.
func TestHelloEmpty(t *testing.T) {
    msg, err := Hello("")
    if msg != "" || err == nil {
        t.Fatalf(`Hello("") = %q, %v, want "", error`, msg, err)
    }
}
// run test: 'go test' or 'go test -v'
```


import akka.actor.{Actor, ActorSystem, Props}
import akka.http.scaladsl.Http
import akka.http.scaladsl.server.Directives._
import akka.pattern.ask
import akka.util.Timeout
import scala.concurrent.duration._
import scala.concurrent.ExecutionContextExecutor

// 1. Define immutable state model
case class BotState(history: List[String] = Nil)
case class ProcessMessage(text: String)
case class GetState()

// 2. Actor to manage state immutably
class BotAgent(initialState: BotState) extends Actor {
  private var state: BotState = initialState

  def receive: Receive = {
    case ProcessMessage(text) =>
      // Create NEW state instead of modifying old data
      state = BotState(text :: state.history)
      sender() ! s"Processed: '$text' | History size: ${state.history.size}"

    case GetState() =>
      sender() ! state
  }
}

// 3. HTTP server to keep bot live
object BotServer extends App {
  implicit val system: ActorSystem = ActorSystem("BotSystem")
  implicit val ec: ExecutionContextExecutor = system.dispatcher
  implicit val timeout: Timeout = Timeout(3.seconds)

  // Initialize with existing data (replace with your actual data)
  val existingData = BotState(List("Old message 1", "Old message 2"))
  val botAgent = system.actorOf(Props(new BotAgent(existingData)), "botAgent")

  val route =
    path("message" / Segment) { text =>
      post {
        complete {
          (botAgent ? ProcessMessage(text))
            .mapTo[String]
            .map(response => s"Bot: $response")
        }
      }
    } ~
    path("state") {
      get {
        complete {
          (botAgent ? GetState())
            .mapTo[BotState]
            .map(state => s"Current state: ${state.history.mkString(", ")}")
        }
      }
    }

  // Start HTTP server (keeps bot live)
  val bindingFuture = Http().newServerAt("0.0.0.0", 8080).bind(route)
  println("Bot running at http://localhost:8080")
}
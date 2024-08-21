import { Component } from '@angular/core';
import { RouterOutlet } from '@angular/router';

import { ClientOverviewComponent } from './components/client-overview/client-overview.component';
import { JobOverviewComponent } from './job-overview/job-overview.component';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [RouterOutlet, ClientOverviewComponent, JobOverviewComponent],
  templateUrl: './app.component.html',
  styleUrl: './app.component.css'
})
export class AppComponent {
  title = 'TradingWeb';
}

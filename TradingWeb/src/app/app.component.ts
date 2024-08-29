import { Component } from '@angular/core';
import { RouterOutlet } from '@angular/router';

import { ClientOverviewComponent } from './components/client-overview/client-overview.component';
import { JobOverviewComponent } from './components/job-overview/job-overview.component';
import { JobConfiguratorComponent } from './components/job-configurator/job-configurator.component';
import { JobCreatorComponent } from './components/job-creator/job-creator.component';
import { UnassignedJobsControlComponent } from './components/unassigned-jobs-control/unassigned-jobs-control.component';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [RouterOutlet, ClientOverviewComponent, JobOverviewComponent,
    JobConfiguratorComponent, JobCreatorComponent, UnassignedJobsControlComponent
  ],
  templateUrl: './app.component.html',
  styleUrl: './app.component.css'
})
export class AppComponent {
  title = 'TradingWeb';
}

import { Component } from '@angular/core';
import { MatListModule } from '@angular/material/list';

@Component({
  selector: 'app-job-overview',
  standalone: true,
  imports: [MatListModule],
  templateUrl: './job-overview.component.html',
  styleUrl: './job-overview.component.css'
})
export class JobOverviewComponent {

}
